#!/bin/env python

import numpy as np
import h5py

import virgo.mpi.parallel_sort as psort
import lustre
import command_line_args
import read_vr

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


def find_matching_halos(cat1_length, cat1_offset, cat1_ids,
                        cat2_length, cat2_offset, cat2_ids,
                        max_nr_particles):
    
    # Find group membership for particles in the first catalogue:
    # Only the first max_nr_particles bound particles in each halo are counted as in a halo.
    cat1_grnr_in_cat1 = read_vr.vr_group_membership_from_ids(cat1_length, cat1_offset, cat1_ids,
                                                             max_nr_particles=max_nr_particles)

    # Find group membership for particles in the second catalogue:
    # In this case all bound particles are considered to be in the halo.
    cat2_grnr_in_cat2 = read_vr.vr_group_membership_from_ids(cat2_length, cat2_offset, cat2_ids)

    # Discard particles which are in no halo from each catalogue
    in_group = (cat1_grnr_in_cat1 >= 0)
    cat1_ids = cat1_ids[in_group]
    cat1_grnr_in_cat1 = cat1_grnr_in_cat1[in_group]
    in_group = (cat2_grnr_in_cat2 >= 0)
    cat2_ids = cat2_ids[in_group]
    cat2_grnr_in_cat2 = cat2_grnr_in_cat2[in_group]

    # For each particle ID in catalogue 1, try to find the same particle ID in catalogue 2
    ptr = psort.parallel_match(cat1_ids, cat2_ids, comm=comm)
    matched = (ptr >= 0)

    # For each particle ID in catalogue 1, fetch the group membership of the matching ID in catalogue 2
    cat1_grnr_in_cat2 = -np.ones_like(cat1_grnr_in_cat1)
    cat1_grnr_in_cat2[matched] = psort.fetch_elements(cat2_grnr_in_cat2, ptr[matched])

    # Discard unmatched particles
    cat1_grnr_in_cat1 = cat1_grnr_in_cat1[matched]
    cat1_grnr_in_cat2 = cat1_grnr_in_cat2[matched]

    # Get sorted, unique (grnr1, grnr2) combinations and counts of how many instances of each we have
    assert np.all(cat1_grnr_in_cat1 < 2**32)
    assert np.all(cat1_grnr_in_cat1 >= 0)
    assert np.all(cat1_grnr_in_cat2 < 2**32)
    assert np.all(cat1_grnr_in_cat2 >= 0)
    sort_key = (cat1_grnr_in_cat1.astype(np.uint64) << 32) + cat1_grnr_in_cat2.astype(np.uint64)
    unique_value, unique_count = psort.parallel_unique(sort_key, comm=comm, return_counts=True)

    # Sort unique combinations by group membership in the first catalogue and then by count.
    cat1_grnr_in_cat1 = unique_value >> 32
    cat1_grnr_in_cat2 = unique_value % (1 << 32)
    assert np.all(unique_count < 2**32)
    sort_key = (cat1_grnr_in_cat1.astype(np.uint64) << 32) + unique_count.astype(np.uint64)
    order = psort.parallel_sort(sort_key, return_index=True, comm=comm)
    cat1_grnr_in_cat1 = psort.fetch_elements(cat1_grnr_in_cat1, order, comm=comm)
    cat1_grnr_in_cat2 = psort.fetch_elements(cat1_grnr_in_cat2, order, comm=comm)
    cat1_count        = psort.fetch_elements(unique_count, order, comm=comm)

    # Zero out the counts for all but the last instance of each group in the first catalogue:
    # The last one has the largest count and its corresponding group index in the second
    # catalogue is the matched halo.
    nr_matches = len(cat1_count)
    colour = 0 if nr_matches > 0 else 1
    comm_nonzero = MPI.COMM_WORLD.Split(colour, comm_rank)
    if nr_matches > 0:
        # Initially assume every instance is the last one
        last = np.ones(len(nr_matches), dtype=bool)
        # Check if the next match on this rank has the same grnr1
        if nr_matches > 1:
            last[cat1_grnr_in_cat1[:-1]==cat1_grnr_in_cat1[1:]] = False
        # If the first match on the next rank has the same grnr1 as our last match
        # then we don't have the last instance of that grnr1
        all_first_grnr1 = comm.allgather(cat1_grnr_in_cat1[0])
        if comm_nonzero.Get_rank() < comm_nonzero.Get_size()-1:
            if all_first_grnr1[comm_nonzero.Get_rank()+1] == cat1_grnr_in_cat1[-1]:
                last[-1] = False
        # Zero the counts
        cat1_count[last==False] = 0

    # Discard entries with zero count
    cat1_grnr_in_cat1 = cat1_grnr_in_cat1[cat1_count > 0]
    cat1_grnr_in_cat2 = cat1_grnr_in_cat2[cat1_count > 0]
    cat1_count = cat1_count[cat1_count > 0]

    
    


if __name__ == "__main__":

    # Read command line parameters
    args = command_line_args.get_match_vr_halos_args(comm)

    # Ensure output dir exists
    if comm_rank == 0:
        lustre.ensure_output_dir(args.output_file)
    comm.barrier()

    # Read VR lengths, offsets and IDs for the two outputs
    (length_bound1, offset_bound1, ids_bound1,
     length_unbound1, offset_unbound1, ids_unbound1) = read_vr.read_vr_lengths_and_offsets(args.vr_basename1)
    (length_bound2, offset_bound2, ids_bound2,
     length_unbound2, offset_unbound2, ids_unbound2) = read_vr.read_vr_lengths_and_offsets(args.vr_basename2)

    # For each halo in output 1, find the matching halo in output 2
    match_index_12 = find_matching_halos(length_bound1, offset_bound1, ids_bound1,
                                         length_bound2, offset_bound2, ids_bound2,
                                         args.nr_particles)

    # For each halo in output 2, find the matching halo in output 1
    match_index_21 = find_matching_halos(length_bound2, offset_bound2, ids_bound2,
                                         length_bound1, offset_bound1, ids_bound1,
                                         args.nr_particles)
    
    # Identify cases where the matches are bijective
    
    # Write the output

