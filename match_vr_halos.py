#!/bin/env python

import numpy as np
import h5py

import virgo.mpi.parallel_sort as psort
import virgo.mpi.parallel_hdf5 as phdf5

import lustre
import command_line_args
import read_vr

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


def message(s):
    if comm_rank == 0:
        print(s)


def exchange_array(arr, dest, comm):
    """
    Carry out an alltoallv on the supplied array, given the MPI rank
    to send each element to.
    """
    order = np.argsort(dest)
    sendbuf = arr[order]
    send_count = np.bincount(dest, minlength=comm_size)
    send_offset = np.cumsum(send_count)-send_count
    recv_count = np.zeros_like(send_count)
    comm.Alltoall(send_count, recv_count)
    recv_offset = np.cumsum(recv_count) - recv_count
    recvbuf = np.ndarray(recv_count.sum(), dtype=arr.dtype)
    psort.my_alltoallv(sendbuf, send_count, send_offset,
                       recvbuf, recv_count, recv_offset,
                       comm=comm)
    return recvbuf


def find_matching_halos(cat1_length, cat1_offset, cat1_ids,
                        cat2_length, cat2_offset, cat2_ids,
                        max_nr_particles):
    
    # Decide range of halos in cat1 which we'll store on each rank:
    # This is used to partition the result between MPI ranks.
    nr_cat1_tot = comm.allreduce(len(cat1_length))
    nr_cat1_per_rank = nr_cat1_tot // comm_size
    if comm_rank < comm_size-1:
        nr_cat1_local = nr_cat1_per_rank
    else:
        nr_cat1_local = nr_cat1_tot - (comm_size-1)*nr_cat1_per_rank

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
    unique_value, cat1_count = psort.parallel_unique(sort_key, comm=comm, return_counts=True, repartition_output=True)
    cat1_grnr_in_cat1 = (unique_value >> 32).astype(int) # Cast to int because mixing signed and unsigned causes numpy to cast to float!
    cat1_grnr_in_cat2 = (unique_value % (1 << 32)).astype(int)

    # Send each (grnr1, grnr2, count) combination to the rank which will store the result for that halo
    dest = (cat1_grnr_in_cat1 // nr_cat1_per_rank).astype(int)
    dest[dest>comm_size-1] = comm_size-1
    recv_grnr_in_cat1 = exchange_array(cat1_grnr_in_cat1, dest, comm)
    recv_grnr_in_cat2 = exchange_array(cat1_grnr_in_cat2, dest, comm)
    recv_count        = exchange_array(cat1_count,        dest, comm)
    
    # Allocate output arrays:
    # Each rank has nr_cat1_per_rank halos with any extras on the last rank
    first_in_cat1 = comm_rank * nr_cat1_per_rank
    result_grnr_in_cat2 = -np.ones(nr_cat1_local, dtype=int)  # For each halo in cat1, will store index of match in cat2
    result_count = np.zeros(nr_cat1_local, dtype=int) # Will store number of matching particles

    # Update output arrays using the received data.
    for recv_nr in range(len(recv_grnr_in_cat1)):
        # Compute local array index of halo to update
        local_halo_nr = recv_grnr_in_cat1[recv_nr] - first_in_cat1
        assert local_halo_nr >=0
        assert local_halo_nr < nr_cat1_local
        # Check if the received count is higher than the highest so far
        if recv_count[recv_nr] > result_count[local_halo_nr]:
            # This received combination has the highest count so far
            result_grnr_in_cat2[local_halo_nr] = recv_grnr_in_cat2[recv_nr]
            result_count[local_halo_nr] = recv_count[recv_nr]
        elif recv_count[recv_nr] == result_count[local_halo_nr]:
            # In the event of a tie, go for the lowest group number for reproducibility
            if recv_grnr_in_cat2[recv_nr] < result_grnr_in_cat2[local_halo_nr]:
                result_grnr_in_cat2[local_halo_nr] = recv_grnr_in_cat2[recv_nr]
                result_count[local_halo_nr] = recv_count[recv_nr]

    return result_grnr_in_cat2, result_count


def consistent_match(match_index_12, match_index_21):
    """
    For each halo in catalogue 1, determine if its match in catalogue 2
    points back at it.

    match_index_12 has one entry for each halo in catalogue 1 and
    specifies the matching halo in catalogue 2 (or -1 for not match)

    match_index_21 has one entry for each halo in catalogue 2 and
    specifies the matching halo in catalogue 1 (or -1 for not match)

    Returns an array with 1 for a match and 0 otherwise.
    """

    # Find the global array indexes of halos stored on this rank
    nr_local_halos = len(match_index_12)
    local_halo_offset = comm.scan(nr_local_halos) - nr_local_halos
    local_halo_index = np.arange(local_halo_offset, local_halo_offset+nr_local_halos, dtype=int)

    # For each halo, find the halo that its match in the other catalogue was matched with
    match_back = -np.ones(nr_local_halos, dtype=int)
    has_match = (match_index_12 >= 0)
    match_back[has_match] = psort.fetch_elements(match_index_21, match_index_12[has_match], comm=comm)
    
    # If we retrieved our own halo index, we have a match
    return np.where(match_back==local_halo_index, 1, 0)


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
    message("Matching from first catalogue to second")
    match_index_12, count_12 = find_matching_halos(length_bound1, offset_bound1, ids_bound1,
                                                   length_bound2, offset_bound2, ids_bound2,
                                                   args.nr_particles)

    # For each halo in output 2, find the matching halo in output 1
    message("Matching from second catalogue to first")
    match_index_21, count_21 = find_matching_halos(length_bound2, offset_bound2, ids_bound2,
                                                   length_bound1, offset_bound1, ids_bound1,
                                                   args.nr_particles)
    
    # Check for consistent matches in both directions
    message("Checking for consistent matches")
    consistent_12 = consistent_match(match_index_12, match_index_21)
    consistent_21 = consistent_match(match_index_21, match_index_12)

    # Write the output
    def write_output_field(name, data, description):
        phdf5.collective_write(outfile, name, data, comm)
        outfile[name].attrs["Description"] = description
    message("Writing output")
    with h5py.File(args.output_file, "w", driver="mpio", comm=comm) as outfile:
        # Write input parameters
        params = outfile.create_group("Parameters")
        for name, value in vars(args).items():
            params.attrs[name] = value
        # Matching from first catalogue to second
        write_output_field("BoundParticleNr1", length_bound1,
                           "Number of bound particles in each halo in the first catalogue")
        write_output_field("MatchIndex1to2", match_index_12,
                           "For each halo in the first catalogue, index of the matching halo in the second")
        write_output_field("MatchCount1to2", count_12,
                           f"How many of the {args.nr_particles} most bound particles from the halo in the first catalogue are in the matched halo in the second")
        write_output_field("Consistent1to2", consistent_12,
                           "Whether the match from first to second catalogue is consistent with second to first (1) or not (0)")
        # Matching from second catalogue to first
        write_output_field("BoundParticleNr2", length_bound2,
                           "Number of bound particles in each halo in the second catalogue")
        write_output_field("MatchIndex2to1", match_index_21,
                           "For each halo in the second catalogue, index of the matching halo in the first")
        write_output_field("MatchCount2to1", count_21,
                           f"How many of the {args.nr_particles} most bound particles from the halo in the second catalogue are in the matched halo in the first")
        write_output_field("Consistent2to1", consistent_21,
                           "Whether the match from second to first catalogue is consistent with first to second (1) or not (0)")
    comm.barrier()
    message("Done.")
