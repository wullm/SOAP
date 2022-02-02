#!/bin/env python3

import numpy as np
import h5py
import astropy.units

import halo_centres
import swift_cells
import so_tasks
import swift_units

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# Location of the input
vr_basename = "/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/vr/catalogue_0013/vr_catalogue_0013.properties"
swift_filename = "/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/snapshots/flamingo_0013.hdf5"
outfile = "./so_props.hdf5"

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
    Mpc = astropy.units.cm * 1e6 * parsec_cgs
    max_halo_radius = 10.0*Mpc
    search_radius = max_halo_radius + 0.5*np.amax(cellgrid.cell_size)

    # Generate task list
    task_list = so_tasks.SOTaskList(cellgrid, so_cat, search_radius=search_radius, cells_per_task=3)

else:
    cellgrid = None

# Make sure all ranks have a copy of the cell grid
cellgrid = comm.bcast(cellgrid)

if comm_rank == 0:

    # Rank 0 responds to requests for tasks
    next_task = 0
    nr_tasks = len(task_list.tasks)
    nr_done = 0
    while nr_done < comm_size-1:
        request_src = comm.recv()
        if next_task < nr_tasks:
            print("Start task %d of %d" % (next_task, nr_tasks))
            comm.send(task_list.tasks[next_task], request_src)
            next_task += 1
        else:
            comm.send(None, request_src)
            nr_done += 1
else:
    
    # Other ranks request and run tasks
    result = []
    while True:
        comm.send(comm_rank, 0)
        task = comm.recv()
        if task is not None:
            result.append(task.run(cellgrid))
        else:
            break

# TODO: use message tags instead of barrier to prevent mix-ups!
comm.barrier()

# Combine results
if comm_rank > 0:

    # Ranks >0 send their lists of results to rank 0
    comm.send(result, 0)

else:
    
    # Rank 0 assembles full result set.
    # First, receive list of results from each other task
    result = []
    for i in range(1, comm_size):
        result += comm.recv(source=i)

    # Then combine into full arrays
    names = result[0].keys()
    all_results = {}
    for name in names:
        all_results[name] = np.concatenate([r[name] for r in result])

    # Sort by halo index
    idx = np.argsort(all_results["index"])
    for name in all_results:
        all_results[name] = all_results[name][idx,...]

    # And write the output file
    with h5py.File(outfile, "w") as outfile:
        for name in all_results:
            outfile[name] = all_results[name]
            if hasattr(all_results[name], "unit"):
                swift_units.write_unit_attributes(outfile[name], all_results[name].unit)
