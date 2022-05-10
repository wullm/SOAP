#!/bin/env python

import numpy as np
import virgo.util.peano as peano

def grid_decomposition(boxsize, centres, nr_chunks):
    """
    Split the simulation volume into a grid of cubic cells.
    Number of chunks must be a cube.

    Returns chunk index for each input halo centre.
    """

    print("Using cubic grid domain decomposition")

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
    Allows an arbitrary number of chunks and tries to put equal
    numbers of halos in each chunk.

    Returns chunk index for each input halo centre.

    TODO: take into account halo masses for load balancing?
          parallelize this?
    
    Will not work well for zoom simulations. Could use a grid
    which just covers the zoom region?
    """

    # Find size of grid to use to calculate PH keys
    bits_per_dimension = 10
    cells_per_dimension = 2**bits_per_dimension
    grid_size = boxsize / cells_per_dimension
    nr_cells = cells_per_dimension**3
    nr_halos = centres.shape[0]
    print(f"Using Peano domain decomposition with bits={bits_per_dimension}")

    # Find integer coordinates of the halos in the cell grid
    ipos = np.floor(centres / grid_size).value.astype(int)
    ipos = np.clip(ipos, 0, cells_per_dimension-1)
    
    # Get PH keys for the halos
    phkey = peano.peano_hilbert_keys(ipos[:,0], ipos[:,1], ipos[:,2], bits_per_dimension)

    # Count cumulative number of halos in each PH cell
    halos_per_cell = np.bincount(phkey, minlength=nr_cells)
    cumulative_halos_per_cell = np.cumsum(halos_per_cell)

    # Decide how many halos we want to put in each chunk
    halos_per_chunk = nr_halos // nr_chunks
    halos_per_chunk = np.ones(nr_chunks, dtype=int) * halos_per_chunk
    halos_per_chunk[:nr_halos % nr_chunks] += 1
    assert np.sum(halos_per_chunk) == nr_halos
    cumulative_halos_per_chunk = np.cumsum(halos_per_chunk) - halos_per_chunk

    # Associate a chunk index with each PH cell
    cell_task_id = -np.ones(nr_cells, dtype=int)
    first_cell_in_chunk = np.searchsorted(cumulative_halos_per_cell, cumulative_halos_per_chunk)
    first_cell_in_chunk[0] = 0
    for i in range(nr_chunks):
        i1 = first_cell_in_chunk[i]
        if i < nr_chunks-1:
            i2 = first_cell_in_chunk[i+1]
        else:
            i2 = nr_cells
        cell_task_id[i1:i2] = i
    assert np.all(cell_task_id >= 0)

    # For each halo, find the chunk index of the PH cell it belongs to
    task_id = cell_task_id[phkey]
    assert np.all(task_id >= 0)
    assert np.all(task_id < nr_chunks)

    return task_id

