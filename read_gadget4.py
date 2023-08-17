#!/bin/env python

import os

import numpy as np
import h5py
import unyt

import virgo.mpi.util
import virgo.mpi.parallel_hdf5 as phdf5


def locate_files(basename):

    snap_format_string = basename+".%(file_nr)d.hdf5"

    # Check that the first snapshot file exists
    snap_file = snap_format_string % {"file_nr" : 0}
    if not(os.path.exists(snap_file)):
        raise IOError("Snapshot file does not exist: "+snap_file)

    # Find the base directory
    topdir = os.path.dirname(os.path.dirname(snap_file))

    # Get the snapshot number from the filename
    snap_nr = int(basename[-3:])

    # Make format string for halo filenames
    group_format_string = f"{topdir}/groups_{snap_nr:03d}/fof_subhalo_tab_{snap_nr:03d}"+".%(file_nr)d.hdf5"

    # Check group file exists
    group_file = group_format_string % {"file_nr" : 0}
    if not(os.path.exists(group_file)):
        raise IOError("Group file does not exist: "+group_file)

    return snap_format_string, group_format_string
    

def read_gadget4_groupnr(basename):
    """
    Read particle IDs and group numbers from Gadget-4 output.

    basename should be the name of a group sorted snapshot file without the
    trailing .N.hdf5.
    """

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Locate the snapshot and fof_subhalo_tab files
    if comm_rank == 0:
        snap_format_string, group_format_string = locate_files(basename)
    else:
        snap_format_string = None
        group_format_string = None
    snap_format_string, group_format_string = comm.bcast((snap_format_string, group_format_string))

    # Check what particle types we have
    if comm_rank == 0:
        snap_file = snap_format_string % {"file_nr" : 0}
        with h5py.File(snap_file, "r") as infile:
            numpart_total = infile["Header"].attrs["NumPart_Total"]
        type_nrs = np.arange(len(numpart_total), dtype=int)
        type_nrs = type_nrs[numpart_total > 0]
    else:
        type_nrs = None
    type_nrs = comm.bcast(type_nrs)
    
    # Read in the sorted particle IDs from the snapshot
    particle_ids = {}
    snap = phdf5.MultiFile(snap_format_string, file_nr_attr=("Header", "NumFilesPerSnapshot"))
    for type_nr in type_nrs:
        particle_ids[type_nr] = snap.read(f"PartType{type_nr}/ParticleIDs")

    # Read in the group lengths and offsets
    subtab = phdf5.MultiFile(group_format_string, file_nr_attr=("Header", "NumFiles"))
    suboffset_type, sublen_type = subtab.read(("Subhalo/SubhaloOffsetType", "Subhalo/SubhaloLenType"), unpack=True)

    # Compute group index for each particle ID
    particle_grnr = {}
    for type_nr in type_nrs:
        particle_grnr[type_nr] = virgo.mpi.util.group_index_from_length_and_offset(np.ascontiguousarray(sublen_type[:,type_nr]),
                                                                                   np.ascontiguousarray(suboffset_type[:,type_nr]),
                                                                                   len(particle_ids[type_nr]),
                                                                                   return_rank=False, comm=comm)    
    # Concatenate and return arrays
    all_grnr = []
    all_ids = []
    for type_nr in type_nrs:
        all_grnr.append(particle_grnr[type_nr])
        all_ids.append(particle_ids[type_nr])

    return np.concatenate(all_ids), np.concatenate(all_grnr)


        
