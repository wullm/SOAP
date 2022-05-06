#!/bin/env python

import numpy as np
import virgo.util.peano as peano

def grid_decomposition(boxsize, centres, nr_chunks):
    """
    Split the simulation volume into a grid of cubic cells.
    Number of chunks must be a cube.

    Returns chunk index for each input halo centre.
    """

    # Determine number of chunks per dimension
    chunks_per_dimension = 1
    while chunks_per_dimension**3 < nr_chunks:
        chunks_per_dimension += 1
    if chunks_per_dimension**3 != nr_chunks:
        raise Exception("Number of chunks must be a cube!")

    # Find size of volume associated with each task
    task_size = boxsize / chunks_per_dimension

    # For each centre, determine integer coords in task grid
    ipos = np.floor(centres / task_size).value.astype(int)
    ipos = np.clip(ipos, 0, chunks_per_dimension-1)

    # Generate a task ID for each halo
    nx = np.amax(ipos[:,0]) + 1
    ny = np.amax(ipos[:,1]) + 1
    nz = np.amax(ipos[:,2]) + 1
    task_id = ipos[:,2] * nx * ny + ipos[:,1] * nx + ipos[:,0] 

    return task_id

    
def peano_decomposition(boxsize, centres, nr_chunks):
    """
    Gadget style domain decomposition using Peano-Hilbert curve.
    Allows arbitrary number of chunks.

    Returns chunk index for each input halo centre.

    TODO: take into account halo masses for load balancing?
          parallelize this?
    """
    
    # Find size of grid to use to calculate PH keys
    bits_per_dimension = 10
    cells_per_dimension = 2**bits_per_dimension
    grid_size = boxsize / cells_per_dimension
    
    # Find integer coordinates of the halos
    ipos = np.floor(centres / grid_size).value.astype(int)
    ipos = np.clip(ipos, 0, cells_per_dimension-1)
    
    # Get PH keys
    phkey = peano.peano_hilbert_keys(ipos[:,0], ipos[:,1], ipos[:,2], bits_per_dimension)

    # Get sorting index of halos by PH key
    order = np.argsort(phkey)

    # Assign halos to chunk tasks:
    # Put a roughly equal number of halos in each chunk.
    nr_halos = centres.shape[0]
    task_id = -np.ones(nr_halos, dtype=int)
    halos_per_chunk = nr_halos // nr_chunks
    for chunk_nr in range(nr_chunks):
        i1 = halos_per_chunk*chunk_nr
        if chunk_nr == nr_chunks-1:
            i2 = nr_halos - 1
        else:
            i2 = i1 + halos_per_chunk
        task_id[order[i1:i2]] = chunk_nr
    assert np.all(task_id >= 0)

    return task_id

