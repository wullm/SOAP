#!/bin/env python

import sys
import os.path
import re
import numpy as np
import h5py

import virgo.mpi.parallel_hdf5
import virgo.mpi.parallel_sort as ps

import lustre
import command_line_args
import read_vr
import read_hbtplus
import read_gadget4

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


if __name__ == "__main__":

    # Read command line parameters
    args = command_line_args.get_group_membership_args(comm)

    # Ensure output dir exists
    if comm_rank == 0:
        lustre.ensure_output_dir(args.output_file)
    comm.barrier()

    # Find group number for each particle ID in the halo finder output
    if args.halo_format == "VR":
        # Read VELOCIraptor output
        (ids_bound, grnr_bound, rank_bound, ids_unbound, grnr_unbound) = read_vr.read_vr_groupnr(args.halo_basename)
        # Read VR host halo IDs, if required. Will set hostHaloID=ID for main halos.
        if args.host_ids:
            host_data = read_vr.read_vr_datasets(args.vr_basename, "properties", ("ID", "hostHaloID",))
            host_id = host_data["hostHaloID"]
            is_main = host_id < 0
            host_id[is_main] = host_data["ID"][is_main]
            del host_data
            del is_main
        else:
            host_id = None
    elif args.halo_format == "HBTplus":
        # Read HBTplus output
        ids_bound, grnr_bound, rank_bound = read_hbtplus.read_hbtplus_groupnr(args.halo_basename)
        ids_unbound = None # HBTplus does not output unbound particles
        grnr_unbound = None
        host_id = None
    elif args.halo_format == "Gadget4":
        # Read Gadget-4 subfind output
        ids_bound, grnr_bound = read_gadget4.read_gadget4_groupnr(args.halo_basename)
        ids_unbound = None
        grnr_unbound = None
        rank_bound = None
        host_id = None
    else:
        raise RuntimeError(f"Unrecognised halo finder name: {args.halo_format}")

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
        if rank_bound is not None:
            swift_rank_bound   = np.ndarray(len(swift_ids), dtype=rank_bound.dtype)
        if ids_unbound is not None:
            swift_grnr_unbound = np.ndarray(len(swift_ids), dtype=grnr_unbound.dtype)

        if comm_rank == 0:
            print("  Matching SWIFT particle IDs to bound IDs")
        ptr = ps.parallel_match(swift_ids, ids_bound)
        
        if comm_rank == 0:
            print("  Assigning bound group membership to SWIFT particles")
        matched = ptr >= 0
        swift_grnr_bound[matched] = ps.fetch_elements(grnr_bound, ptr[matched])
        swift_grnr_bound[matched==False] = -1

        if rank_bound is not None:
            if comm_rank == 0:
                print("  Assigning rank by binding energy to SWIFT particles")
            swift_rank_bound[matched] = ps.fetch_elements(rank_bound, ptr[matched])
            swift_rank_bound[matched==False] = -1

        if ids_unbound is not None:
            if comm_rank == 0:
                print("  Matching SWIFT particle IDs to unbound IDs")
            ptr = ps.parallel_match(swift_ids, ids_unbound)

            if comm_rank == 0:
                print("  Assigning unbound group membership to SWIFT particles")
            matched = ptr >= 0
            swift_grnr_unbound[matched] = ps.fetch_elements(grnr_unbound, ptr[matched])
            swift_grnr_unbound[matched==False] = -1
            swift_grnr_all = np.maximum(swift_grnr_bound, swift_grnr_unbound)

        if host_id is not None:
            if comm_rank == 0:
                print("  Assigning host halo membership to SWIFT particles")
            swift_hostnr_all = -np.ones_like(swift_grnr_all)
            in_halo = swift_grnr_all >= 0
            swift_hostnr_all[in_halo] = ps.fetch_elements(host_id, swift_grnr_all[in_halo], comm=comm) - 1
        else:
            swift_hostnr_all = None
            
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
        attrs = {
            "GroupNr_bound" : {"Description" : "Index of halo in which this particle is a bound member, or -1 if none"},
            "Rank_bound" : {"Description" : "Ranking by binding energy of the bound particles (first in halo=0), or -1 if not bound"},
            "GroupNr_all" : {"Description" : "Index of halo in which this particle is a member (bound or unbound), or -1 if none"},
            "HostNr_all" : {"Description" : "Index of the host of the halo which this particle belongs to (bound or unbound)"},
        }
        attrs["GroupNr_bound"].update(unit_attrs)
        attrs["Rank_bound"].update(unit_attrs)
        attrs["GroupNr_all"].update(unit_attrs)
        attrs["HostNr_all"].update(unit_attrs)

        # Write these particles out with the same layout as the input snapshot
        if comm_rank == 0:
            print("  Writing out group membership of SWIFT particles")
        elements_per_file = snap_file.get_elements_per_file("ParticleIDs", group=ptype)
        output = {"GroupNr_bound"   : swift_grnr_bound}
        if rank_bound is not None:
            output["Rank_bound"] = swift_rank_bound
        if ids_unbound is not None:
            output["GroupNr_all"] = swift_grnr_all
        if host_id is not None:
            output["HostNr_all"] = swift_hostnr_all
        snap_file.write(output, elements_per_file, filenames=args.output_file, mode=mode, group=ptype, attrs=attrs)

    # Optionally, also make a virtual snapshot with group membership information
    if args.virtual_snapshot is not None and comm_rank == 0:

        # Find the original virtual snapshot created by SWIFT
        virtual_snapshot = (args.swift_filename % {"file_nr" : 0})[:-7]+".hdf5"

        # Make a new virtual snapshot file containing group membership information
        from make_virtual_snapshot import make_virtual_snapshot
        make_virtual_snapshot(virtual_snapshot, args.output_file, args.virtual_snapshot)

        # Add absolute paths to the datasets in the virtual file:
        # This is necessary because we can't set two different VDS prefixes.
        from update_vds_paths import update_virtual_snapshot_paths
        snapshot_dir = os.path.abspath(os.path.dirname(args.swift_filename % {"file_nr" : 0}))
        membership_dir = os.path.abspath(os.path.dirname(args.output_file % {"file_nr" : 0}))
        update_virtual_snapshot_paths(args.virtual_snapshot, snapshot_dir, membership_dir)

    comm.barrier()
    if comm_rank == 0:
        print("Done.")

