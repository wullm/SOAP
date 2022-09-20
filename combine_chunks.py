#!/bin/env python

import numpy as np
import h5py

import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort

import swift_units


def sub_snapnum(filename, snapnum):
    """
    Substitute the snapshot number into a filename format string
    without substituting the file number.
    """
    filename = filename.replace("%(file_nr)", "%%(file_nr)")
    return filename % {"snap_nr": snapnum}


def combine_chunks(args, cellgrid, halo_prop_list, scratch_file_format,
                   ref_metadata, nr_chunks, comm_world):
    """
    Combine the per-chunk output files into a single, sorted output
    """

    # First MPI rank creates the output file and writes some metadata in serial mode
    output_file = sub_snapnum(args.output_file, args.snapshot_nr)
    if comm_world.Get_rank() == 0:
        outfile = h5py.File(output_file, "w")
        cellgrid.write_metadata(outfile.create_group("SWIFT"))
        params = outfile.create_group("Parameters")
        params.attrs["swift_filename"] = args.swift_filename
        params.attrs["vr_basename"] = args.vr_basename
        params.attrs["snapshot_nr"] = args.snapshot_nr
        params.attrs["centrals_only"] = 0 if args.centrals_only == False else 1
        calc_names = sorted([hp.name for hp in halo_prop_list])
        params.attrs["calculations"] = calc_names
        params.attrs["halo_ids"] = (
            args.halo_ids if args.halo_ids is not None else np.ndarray(0, dtype=int)
        )
        outfile.close()

    # Open the per-chunk scratch files
    scratch_file = phdf5.MultiFile(
        scratch_file_format, file_idx=range(nr_chunks), comm=comm_world
    )

    # Read the VR halo IDs from the scratch files and make a sorting index to put them in order
    vr_id = scratch_file.read(("VR/ID",))["VR/ID"]
    order = psort.parallel_sort(vr_id, return_index=True, comm=comm_world)
    del vr_id

    # Reopen the output file in parallel mode
    outfile = h5py.File(output_file, "r+", driver="mpio", comm=comm_world)

    # Loop over halo properties, a few at a time
    total_nr_props = len(ref_metadata)
    props_per_iteration = min(total_nr_props, 100)  # TODO: how to choose this number?
    for i1 in range(0, total_nr_props, props_per_iteration):
        i2 = min(i1 + props_per_iteration, total_nr_props)

        # Find the properties to reorder on this iteration
        names, sizes, units, dtypes, descriptions = zip(*ref_metadata[i1:i2])

        # Read in and reorder the properties
        data = scratch_file.read(names)
        for name in names:
            data[name] = psort.fetch_elements(data[name], order, comm=comm_world)

        # Write these properties to the output file
        for name, size, unit, description in zip(names, sizes, units, descriptions):
            # Write the data
            phdf5.collective_write(outfile, name, data[name], comm=comm_world)
            # Add units and description
            attrs = swift_units.attributes_from_units(unit)
            attrs["Description"] = description
            for attr_name, attr_value in attrs.items():
                outfile[name].attrs[attr_name] = attr_value

        del data

    outfile.close()
