#!/bin/env python

import numpy as np
import h5py

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


def hbt_filename(hbt_basename, file_nr):
    return f"{hbt_basename}.{file_nr}.hdf5"


def read_hbtplus_groupnr(basename):
    """
    Read HBTplus output and return group number for each particle ID
    """
    # Find number of HBT output files
    if comm_rank == 0:
        with h5py.File(hbt_filename(basename, 0), "r") as infile:
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
        with h5py.File(hbt_filename(basename, file_nr), "r") as infile:
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

    return ids_bound, grnr_bound, rank_bound
