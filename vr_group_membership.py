#!/bin/env python

import sys
import re
import numpy as np
import h5py

import virgo.mpi.util
import virgo.mpi.parallel_hdf5
import virgo.mpi.gather_array as ga
import virgo.mpi.parallel_sort as ps

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


def read_vr_datasets(vr_basename, file_type, datasets, return_file_nr=None):
    """
    Parallel read of datasets from VR split over any number of files
    """
    # Make a format string for the filename
    filename_format = vr_basename+"."+file_type+".%(file_nr)d"

    # Open the file
    vr_file = virgo.mpi.parallel_hdf5.MultiFile(filename_format, file_nr_dataset="Num_of_files")

    # Read the data
    return vr_file.read(datasets, return_file_nr=return_file_nr)


def compute_lengths(offsets, total_nr_ids):
    """
    Compute group lengths given the offsets and the total number
    of particle IDs. 
    """
    
    # Only include ranks with >0 groups
    if len(offsets) > 0:
        color = 1
    else:
        color = 0
    local_comm = comm.Split(color)

    if color==1:
        # Find first offset on each rank
        first_offset = local_comm.allgather(offsets[0])
        # Allocate lengths array
        lengths = np.ndarray(len(offsets), dtype=int)
        # Compute lengths of all but last group
        lengths[:-1] = offsets[1:] - offsets[:-1]
        # Compute length of last group
        if local_comm.Get_rank() == local_comm.Get_size()-1:
            lengths[-1] = total_nr_ids - offsets[-1]
        else:
            lengths[-1] = first_offset[local_comm.Get_rank()+1] - offsets[-1]
    else:
        # Have zero groups on this rank
        lengths = np.ndarray(0, dtype=int)

    local_comm.Free()
    return lengths


def find_group_membership(vr_basename):
    """
    For each particle ID in a VR output, find the halo index
    """

    # Find number of VR output files
    fname = vr_basename+".catalog_particles.0"
    if comm_rank == 0:
        infile = h5py.File(fname, "r")
        nr_files = infile["Num_of_files"][0]
        infile.close()
    else:
        nr_files = None
    nr_files = comm.bcast(nr_files)

    # Assign files to ranks
    files_on_rank = virgo.mpi.parallel_hdf5.assign_files(nr_files, comm_size)    
    first_file = np.cumsum(files_on_rank) - files_on_rank

    # Loop over files on this rank and read numbers of bound, unbound IDs
    nr_ids_bound = []
    nr_ids_unbound = []
    for file_nr in range(first_file[comm_rank], first_file[comm_rank]+files_on_rank[comm_rank]):
        fname = vr_basename+(".catalog_particles.%d" % file_nr)
        with h5py.File(fname, "r") as infile:
            nr_ids_bound.append(infile["Particle_IDs"].shape[0])
        fname = vr_basename+(".catalog_particles.unbound.%d" % file_nr)
        with h5py.File(fname, "r") as infile:
            nr_ids_unbound.append(infile["Particle_IDs"].shape[0])

    # Combine results from all ranks
    nr_ids_bound = np.concatenate(comm.allgather(np.asarray(nr_ids_bound, dtype=int)), dtype=int)
    nr_ids_unbound = np.concatenate(comm.allgather(np.asarray(nr_ids_unbound, dtype=int)), dtype=int)

    # Find offsets which need to be added to particle offsets read from each file
    cumulative_nr_bound_ids = np.cumsum(nr_ids_bound) - nr_ids_bound
    cumulative_nr_unbound_ids = np.cumsum(nr_ids_unbound) - nr_ids_unbound

    # Read in the VR particle offsets
    names = ("Offset","Offset_unbound")
    data, file_nr = read_vr_datasets(vr_basename, "catalog_groups", names, return_file_nr=names)

    # Make all offsets relative to the start of file zero
    offset_bound = data["Offset"] + cumulative_nr_bound_ids[file_nr["Offset"]]
    offset_unbound = data["Offset_unbound"] + cumulative_nr_unbound_ids[file_nr["Offset_unbound"]]

    # Report number of groups read in
    nr_bound_offsets = comm.reduce(len(offset_bound))
    nr_unbound_offsets = comm.reduce(len(offset_unbound))
    if comm_rank == 0:
        print("Read in %d bound offsets and %d unbound offsets" % (nr_bound_offsets, nr_unbound_offsets))

    # Read in the particle IDs
    data = read_vr_datasets(vr_basename, "catalog_particles", ("Particle_IDs",))
    ids_bound = data["Particle_IDs"]
    data = read_vr_datasets(vr_basename, "catalog_particles.unbound", ("Particle_IDs",))
    ids_unbound = data["Particle_IDs"]
    nr_bound_ids = comm.reduce(len(ids_bound))
    nr_unbound_ids = comm.reduce(len(ids_unbound))
    if comm_rank == 0:
        print("Read in %d bound ids and %d unbound ids" % (nr_bound_ids, nr_unbound_ids))

    # Find the bound and unbound length of each halo
    total_nr_bound = comm.allreduce(len(ids_bound))
    length_bound = compute_lengths(offset_bound, total_nr_bound)
    total_nr_unbound = comm.allreduce(len(ids_unbound))
    length_unbound = compute_lengths(offset_unbound, total_nr_unbound)
    if comm_rank == 0:
        print("Calculated halo lengths ")

    # Associate a group index to each particle ID
    grnr_bound = virgo.mpi.util.group_index_from_length_and_offset(length_bound, offset_bound, len(ids_bound))
    grnr_unbound = virgo.mpi.util.group_index_from_length_and_offset(length_unbound, offset_unbound, len(ids_unbound))

    return ids_bound, grnr_bound, ids_unbound, grnr_unbound


if __name__ == "__main__":

    args = {}
    if comm_rank == 0:
        args["swift_filename"] = sys.argv[1] # Name of one snapshot file
        args["vr_basename"]    = sys.argv[2] # Name of VR files, minus the trailing .filetype.N
        args["outfile"]        = sys.argv[3] # Name of the output file
    args = comm.bcast(args)

    # Find group number for each particle ID in the VR output
    ids_bound, grnr_bound, ids_unbound, grnr_unbound = find_group_membership(args["vr_basename"])

    # Determine SWIFT particle types which exist in the snapshot
    ptypes = []
    with h5py.File(args["swift_filename"], "r") as infile:
        nr_types = infile["Header"].attrs["NumPartTypes"][0]
        numpart_total = (infile["Header"].attrs["NumPart_Total"].astype(np.int64) +
                         infile["Header"].attrs["NumPart_Total_HighWord"].astype(np.int64) << 32)
        nr_files = infile["Header"].attrs["NumFilesPerSnapshot"][0]
        for i in range(nr_types):
            if numpart_total[i] > 0:
                ptypes.append("PartType%d" % i)

    # Find format string for SWIFT snapshot file names
    if nr_files > 1:
        m = re.match(r"(.*)\.[0-9]+\.hdf5", args["swift_filename"])
        if m is not None:
            swift_filename_fmt = m.group(1)+".%(file_nr)d.hdf5"
        else:
            raise ValueError("Don't understand SWIFT filename")
    else:
        swift_filename_fmt = args["swift_filename"]

    # Open the snapshot
    snap_file = virgo.mpi.parallel_hdf5.MultiFile(swift_filename_fmt,
                                                  file_nr_attr=("Header", "NumFilesPerSnapshot"))

    # Loop over particle types
    create_file = True
    for ptype in ptypes:

        if comm_rank == 0:
            print("Calculating group membership for type ", ptype)
        swift_ids = snap_file.read(("ParticleIDs",), ptype)["ParticleIDs"]

        # Allocate array to store SWIFT particle group membership
        swift_grnr_bound   = np.ndarray(len(swift_ids), dtype=grnr_bound.dtype)
        swift_grnr_unbound = np.ndarray(len(swift_ids), dtype=grnr_bound.dtype)

        if comm_rank == 0:
            print("  Matching SWIFT particle IDs to VR bound IDs")
        ptr = ps.parallel_match(swift_ids, ids_bound)
        
        if comm_rank == 0:
            print("  Assigning VR bound group membership to SWIFT particles")
        matched = ptr >= 0
        swift_grnr_bound[matched] = ps.fetch_elements(grnr_bound, ptr[matched])
        swift_grnr_bound[matched==False] = -1

        if comm_rank == 0:
            print("  Matching SWIFT particle IDs to VR unbound IDs")
        ptr = ps.parallel_match(swift_ids, ids_unbound)
        
        if comm_rank == 0:
            print("  Assigning VR unbound group membership to SWIFT particles")
        matched = ptr >= 0
        swift_grnr_unbound[matched] = ps.fetch_elements(grnr_unbound, ptr[matched])
        swift_grnr_unbound[matched==False] = -1

        # Determine if we need to create a new output file set
        if create_file:
            mode="w"
            create_file=False
        else:
            mode="r+"

        # Write these particles out with the same layout as the snapshot
        if comm_rank == 0:
            print("  Writing out VR group membership of SWIFT particles")
        elements_per_file = snap_file.get_elements_per_file("ParticleIDs", group=ptype)
        output = {"GroupNr_bound"   : swift_grnr_bound,
                  "GroupNr_unbound" : swift_grnr_unbound}
        snap_file.write(output, elements_per_file, filenames=args["outfile"], mode=mode, group=ptype)

    comm.barrier()
    if comm_rank == 0:
        print("Done.")
    
