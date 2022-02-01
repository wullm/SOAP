#!/bin/env python3

import numpy as np

import halo_centres
import swift_cells
import so_tasks
import astropy.units

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# Location of the input
vr_basename = "/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/vr/catalogue_0013/vr_catalogue_0013.properties"
swift_filename = "/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/snapshots/flamingo_0013.hdf5"

# Rank zero reads the halo positions and generates a list of tasks
if comm_rank == 0:

    # Read SWIFT cells
    cellgrid = swift_cells.SWIFTCellGrid(swift_filename)
    parsec_cgs = cellgrid.constants["parsec"]
    solar_mass_cgs = cellgrid.constants["solar_mass"]
    a = cellgrid.a

    # Read the halo catalogue
    so_cat = halo_centres.SOCatalogue(vr_basename, a, parsec_cgs, solar_mass_cgs)

    # Decide on search radius
    mpc = astropy.units.cm * parsec_cgs
    max_halo_radius = 10.0*mpc
    search_radius = max_halo_radius + 0.5*np.amax(cellgrid.cell_size)*mpc

    # Generate task list
    task_list = so_tasks.SOTaskList(cellgrid, so_cat, search_radius=search_radius, cells_per_task=3)

else:
    cellgrid = None

# Make sure all ranks have a copy of the cell grid
cellgrid = comm.bcast(cellgrid)

if comm_rank == 0:

    # Rank 0 responds to requests for tasks
    next_task = 0
    nr_tasks = 1 #len(task_list.tasks) # Only do one task for now
    nr_done = 0
    while nr_done < comm_size-1:
        request_src = comm.recv()
        if next_task < nr_tasks:
            comm.send(task_list.tasks[next_task], request_src)
            next_task += 1
        else:
            comm.send(None, request_src)
            nr_done += 1
else:
    
    # Other ranks request and run tasks
    while True:
        comm.send(comm_rank, 0)
        task = comm.recv()
        if task is not None:
            task.run(cellgrid)
        else:
            break

# Then gather and output results...

