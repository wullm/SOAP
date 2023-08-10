#!/bin/env python

import sys
import re
import numpy as np
import h5py

import virgo.mpi.parallel_hdf5
import virgo.mpi.parallel_sort as ps

import lustre
import command_line_args

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


def hbt_filename(hbt_basename, file_nr):
    return f"{hbt_basename}.{file_nr}.hdf5"


if __name__ == "__main__":

    # Read command line parameters
    args = command_line_args.get_hbt_group_membership_args(comm)

    # Ensure output dir exists
    if comm_rank == 0:
        lustre.ensure_output_dir(args.output_file)
    comm.barrier()

    # Find number of HBT output files
    if comm_rank == 0:
        with h5py.File(hbt_filename(args.hbt_basename, 0), "r") as infile:
            nr_files = int(infile["NumberOfFiles"][...])
    else:
        nr_files = None
    nr_files = comm.bcast(nr_files)

    # Assign files to MPI ranks
    files_per_rank = np.zeros(comm_size, dtype=int)
    files_per_rank[:] = nr_files // comm_size
    files_per_rank[:nr_files % comm_size] += 1
    assert np.sum(files_per_rank) == nr_files
    first_file_on_rank = np.cumsum(files_per_rank) - files_per_rank

    # Read in the halos from the HBT output:
    # 'halos' will be an array of structs with the halo catalogue
    # 'ids_bound' will be an array of particle IDs in halos, sorted by halo
    halos = []
    ids_bound = []
    for file_nr in range(first_file_on_rank[comm_rank],
                         first_file_on_rank[comm_rank]+files_per_rank[comm_rank]):
        with h5py.File(hbt_filename(args.hbt_basename, file_nr), "r") as infile:
            halos.append(infile["Subhalos"][...])
            ids_bound.append(infile["SubhaloParticles"][...])
    halos = np.concatenate(halos)
    ids_bound = np.concatenate(ids_bound) # Combine arrays from different files
    ids_bound = np.concatenate(ids_bound) # Combine arrays from different halos
    
    # Assign halo indexes to the particles
    nr_local_halos = len(halos)
    halo_offset = comm.scan(len(halos), op=MPI.SUM) - len(halos)
    halo_index = np.arange(nr_local_halos, dtype=int) + halo_offset
    halo_size = halos["Nbound"]
    grnr_bound = np.repeat(halo_index, halo_size)

    # Assign ranking by binding energy to the particles
    rank_bound = -np.ones(grnr_bound.shape[0], dtype=int)
    offset = 0
    for halo_nr in range(nr_local_halos):
        rank_bound[offset:offset+halo_size[halo_nr]] = np.arange(halo_size[halo_nr], dtype=int)
        offset += halo_size[halo_nr]
    assert np.all(rank_bound >= 0) # HBT only outputs bound particles

    # Determine SWIFT particle types which exist in the snapshot
    ptypes = []
    with h5py.File(args.swift_filename % {"file_nr" : 0}, "r") as infile:
        nr_types = infile["Header"].attrs["NumPartTypes"][0]
        numpart_total = (infile["Header"].attrs["NumPart_Total"].astype(np.int64) +
                         (infile["Header"].attrs["NumPart_Total_HighWord"].astype(np.int64) << 32))
        nr_files = infile["Header"].attrs["NumFilesPerSnapshot"][0]
        for i in range(nr_types):
            if numpart_total[i] > 0:
                ptypes.append("PartType%d" % i)

    # Open the snapshot
    snap_file = virgo.mpi.parallel_hdf5.MultiFile(args.swift_filename,
                                                  file_nr_attr=("Header", "NumFilesPerSnapshot"))

    # Loop over particle types
    create_file = True
    for ptype in ptypes:

        if comm_rank == 0:
            print("Calculating group membership for type ", ptype)
        swift_ids = snap_file.read(("ParticleIDs",), ptype)["ParticleIDs"]

        # Allocate array to store SWIFT particle group membership
        swift_grnr_bound   = np.ndarray(len(swift_ids), dtype=grnr_bound.dtype)
        swift_rank_bound   = np.ndarray(len(swift_ids), dtype=rank_bound.dtype)

        if comm_rank == 0:
            print("  Matching SWIFT particle IDs to bound IDs")
        ptr = ps.parallel_match(swift_ids, ids_bound)
        
        if comm_rank == 0:
            print("  Assigning bound group membership to SWIFT particles")
        matched = ptr >= 0
        swift_grnr_bound[matched] = ps.fetch_elements(grnr_bound, ptr[matched])
        swift_grnr_bound[matched==False] = -1

        if comm_rank == 0:
            print("  Assigning rank by binding energy to SWIFT particles")
        swift_rank_bound[matched] = ps.fetch_elements(rank_bound, ptr[matched])
        swift_rank_bound[matched==False] = -1

        # Determine if we need to create a new output file set
        if create_file:
            mode="w"
            create_file=False
        else:
            mode="r+"

        # Set up dataset attributes
        unit_attrs = {
            "Conversion factor to CGS (not including cosmological corrections)" : [1.0,],
            "Conversion factor to CGS (including cosmological corrections)" : [1.0,],
            "U_I exponent" : [0.0,],
            "U_L exponent" : [0.0,],
            "U_M exponent" : [0.0,],
            "U_t exponent" : [0.0,],
            "U_T exponent" : [0.0,],
            "a-scale exponent" : [0.0,],
            "h-scale exponent" : [0.0,],
        }
        attrs = {"GroupNr_bound" : {"Description" : "Index of halo in which this particle is a bound member, or -1 if none"},
                 "Rank_bound" : {"Description" : "Ranking by binding energy of the bound particles (first in halo=0), or -1 if not bound"},}
        attrs["GroupNr_bound"].update(unit_attrs)
        attrs["Rank_bound"].update(unit_attrs)

        # Write these particles out with the same layout as the input snapshot
        if comm_rank == 0:
            print("  Writing out group membership of SWIFT particles")
        elements_per_file = snap_file.get_elements_per_file("ParticleIDs", group=ptype)
        output = {"GroupNr_bound"   : swift_grnr_bound,
                  "Rank_bound"      : swift_rank_bound,}
        snap_file.write(output, elements_per_file, filenames=args.output_file, mode=mode, group=ptype, attrs=attrs)

        # Optionally, also write the particle group membership to the specified single file snapshot.
        # (e.g. a copy of the virtual file written by SWIFT)
        if args.update_virtual_file is not None:
            prefix = args.output_prefix if args.output_prefix is not None else ""
            if comm_rank == 0:
                print("  Writing out group membership to virtual file")
            vfile = h5py.File(args.update_virtual_file, "r+", driver="mpio", comm=comm)
            virgo.mpi.parallel_hdf5.collective_write(vfile[ptype], prefix+"GroupNr_bound", swift_grnr_bound, comm)
            virgo.mpi.parallel_hdf5.collective_write(vfile[ptype], prefix+"Rank_bound", swift_rank_bound, comm)
            vfile.close()

    comm.barrier()
    if comm_rank == 0:
        print("Done.")

