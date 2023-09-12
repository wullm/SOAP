#!/bin/env python

import os.path
import numpy as np
import h5py
import unyt

import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort


def hbt_filename(hbt_basename, file_nr):
    return f"{hbt_basename}.{file_nr}.hdf5"


def read_hbtplus_groupnr(basename):
    """
    Read HBTplus output and return group number for each particle ID
    """

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

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

    # Now check for duplicates. HBTplus can assign the same particle to multiple
    # subhalos in cases where a subhalo escapes its host and enters another halo.
    # Here we assign such particles to the subhalo in which their rank_bound is lowest.
    # Make a key to sort the particles by ID and then by bound rank where ID is equal.
    sort_key_t = np.dtype([("id", ids_bound.dtype), ("rank", rank_bound.dtype)])
    sort_key = np.ndarray(len(ids_bound), dtype=sort_key_t)
    sort_key["id"] = ids_bound
    sort_key["rank"] = rank_bound
    assert np.all(sort_key["rank"] >= 0) # HBTplus does not output unbound particles

    # Sort the particles by ID and then by bound rank where the ID is the same.
    order = psort.parallel_sort(sort_key, return_index=True, comm=comm)
    del sort_key
    ids_bound  = psort.fetch_elements(ids_bound,  order, comm=comm)    
    grnr_bound = psort.fetch_elements(grnr_bound, order, comm=comm)
    rank_bound = psort.fetch_elements(rank_bound, order, comm=comm)
    del order
    
    # Then find the unique particle IDs
    unique_ids_bound, unique_counts = psort.parallel_unique(ids_bound, comm=comm, arr_sorted=True,return_counts=True)
    nr_unique_ids_local = len(unique_ids_bound)
    
    # Find out how many unique IDs are on each previous MPI rank
    nr_unique_ids_prev_rank = comm.scan(nr_unique_ids_local) - nr_unique_ids_local
    
    # Find the global offset of the first instance of each ID
    unique_offsets = np.cumsum(unique_counts) - unique_counts + nr_unique_ids_prev_rank
    
    # Fetch the ID, grnr_bound and rank_bound of the first instance of each particle ID
    ids_bound  = psort.fetch_elements(ids_bound,  unique_offsets, comm=comm)
    assert(np.all(ids_bound)==unique_ids_bound) # Check we computed unique_offsets correctly
    rank_bound = psort.fetch_elements(rank_bound, unique_offsets, comm=comm)
    grnr_bound = psort.fetch_elements(grnr_bound, unique_offsets, comm=comm)
    
    return ids_bound, grnr_bound, rank_bound


def read_hbtplus_catalogue(comm, basename, a_unit, registry, boxsize):
    """
    Read in the HBTplus halo catalogue, distributed over communicator comm.

    comm     - communicator to distribute catalogue over
    basename - HBTPlus SubSnap filename without the .N suffix
    a_unit   - unyt a factor
    registry - unyt unit registry
    boxsize  - box size as a unyt quantity

    Returns a dict of unyt arrays with the halo properies.
    Arrays which must always be returned:

    index - index of each halo in the input catalogue
    cofp  - (N,3) array with centre to use for SO calculations
    search_radius - initial search radius which includes all member particles
    is_central - integer 1 for centrals, 0 for satellites
    nr_bound_part - number of bound particles in each halo
    
    Any other arrays will be passed through to the output ONLY IF they are
    documented in property_table.py.

    Note that in case of HBT we only want to compute properties of resolved
    halos, so we discard those with 0-1 bound particles.
    """

    comm_size = comm.Get_size()    
    comm_rank = comm.Get_rank()
    
    # Get SWIFT's definition of physical and comoving Mpc units
    swift_pmpc = unyt.Unit("swift_mpc",       registry=registry)
    swift_cmpc = unyt.Unit(a_unit*swift_pmpc, registry=registry)
    swift_msun = unyt.Unit("swift_msun",      registry=registry)
    
    # Get expansion factor as a float
    a = a_unit.base_value

    # Get h as a float
    h_unit = unyt.Unit("h", registry=registry)
    h = h_unit.base_value
    
    # Get HBTplus unit information
    if comm_rank == 0:
        filename = hbt_filename(basename, 0)
        with h5py.File(filename, "r") as infile:
            LengthInMpch = float(infile["Units/LengthInMpch"][...])
            MassInMsunh  = float(infile["Units/MassInMsunh"][...])
            VelInKmS     = float(infile["Units/VelInKmS"][...])
    else:
        LengthInMpch = None
        MassInMsunh  = None
        VelInKmS     = None
    (LengthInMpch, MassInMsunh, VelInKmS) = comm.bcast((LengthInMpch, MassInMsunh, VelInKmS))

    # Read the subhalos for this snapshot
    filename = f"{basename}.%(file_nr)d.hdf5"
    mf = phdf5.MultiFile(filename, file_nr_dataset="NumberOfFiles", comm=comm)
    subhalo = mf.read("Subhalos")

    # Only process resolved subhalos (HBTplus also outputs unresolved "orphan" subhalos)
    keep = subhalo["Nbound"] > 1
    
    # Assign indexes to halos: for each halo we're going to process we store the
    # position in the input catalogue.
    nr_local_halos = len(keep)
    local_offset = comm.scan(nr_local_halos) - nr_local_halos
    index = np.arange(nr_local_halos, dtype=int) + local_offset
    index = index[keep]
    index = unyt.unyt_array(index, units=unyt.dimensionless, dtype=int, registry=registry)
        
    # Find centre of potential
    cofp = (subhalo["ComovingMostBoundPosition"][keep,:] * LengthInMpch / h) * swift_cmpc

    # Initial guess at search radius for each halo - twice the half mass radius.
    # Search radius will be expanded if we don't find all of the bound particles.
    search_radius = 2.0*(subhalo["RHalfComoving"][keep] * LengthInMpch / h) * swift_cmpc

    # Central halo flag
    is_central = np.where(subhalo["Rank"]==0, 1, 0)[keep]
    is_central = unyt.unyt_array(is_central, units=unyt.dimensionless, dtype=int, registry=registry)
    
    # Number of bound particles
    nr_bound_part = subhalo["Nbound"][keep]
    nr_bound_part = unyt.unyt_array(nr_bound_part, units=unyt.dimensionless, dtype=int, registry=registry)

    local_halo = {
        "cofp"          : cofp,
        "index"         : index,
        "search_radius" : search_radius,
        "is_central"    : is_central,
        "nr_bound_part" : nr_bound_part,
    }
    return local_halo
