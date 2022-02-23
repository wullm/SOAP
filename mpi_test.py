#!/bin/env python

import sys
import threading
import time

import numpy as np
import h5py
import astropy.units

import halo_centres
import swift_cells
import so_tasks
import swift_units

import halo_properties

# Initialize mpi4py with thread support
import mpi4py
mpi4py.rc.threads=True
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# Message tags required to prevent mix-ups!
REQUEST_TASK_TAG=1
ASSIGN_TASK_TAG=2
RETURN_RESULT_TAG=3


def sleepy_recv(tag):
    """
    Wait for a message without keeping a core spinning so that we leave
    the core available to run jobs and release the GIL. Checks for
    incoming messages at exponentially increasing intervals starting
    at 1.0e-6s up to a limit of ~1s. Sleeps between checks.
    """
    request = comm.irecv(tag=tag)
    delay = 1.0e-6
    while True:
        completed, message = request.test()
        if completed:
            return message
        if delay < 1.0:
            delay *= 2.0
        time.sleep(delay)


def distribute_tasks(task_list):
    """
    Listen for and respond to requests for tasks to do
    """
    next_task = 0
    nr_tasks = len(task_list.tasks)
    nr_done = 0
    while nr_done < comm_size:
        request_src = sleepy_recv(tag=REQUEST_TASK_TAG)
        if next_task < nr_tasks:
            print("Starting task %d of %d" % (next_task, nr_tasks))
            comm.send(task_list.tasks[next_task], request_src, tag=ASSIGN_TASK_TAG)
            next_task += 1
        else:
            comm.send(None, request_src, tag=ASSIGN_TASK_TAG)
            nr_done += 1
            print("Number of ranks done with all tasks = %d" % nr_done)
    print("All tasks done.")


if __name__ == "__main__":

    # Read command line parameters
    args = {}
    if comm_rank == 0:
        args["swift_filename"] = sys.argv[1] # Name of one snapshot file
        args["vr_basename"]    = sys.argv[2] # Name of properties file, minus the trailing .N
        args["cells_per_task"] = int(sys.argv[3]) # 1D size of each task in top level cells
        args["outfile"]        = sys.argv[4] # Name of the output file
    args = comm.bcast(args)

    # Make a list of properties to calculate
    halo_prop_list = [halo_properties.SOMasses(),]

    # Rank zero reads the halo positions and generates a list of tasks
    if comm_rank == 0:

        # Read SWIFT cells
        cellgrid = swift_cells.SWIFTCellGrid(args["swift_filename"])
        parsec_cgs = cellgrid.constants["parsec"]
        solar_mass_cgs = cellgrid.constants["solar_mass"]
        a = cellgrid.a

        # Read the halo catalogue
        so_cat = halo_centres.SOCatalogue(args["vr_basename"], a, parsec_cgs, solar_mass_cgs)

        # Decide on search radius
        Mpc = astropy.units.cm * 1e6 * parsec_cgs
        max_halo_radius = 10.0*Mpc
        search_radius = max_halo_radius + 0.5*np.amax(cellgrid.cell_size)

        # Generate task list
        task_list = so_tasks.SOTaskList(cellgrid, so_cat, search_radius=search_radius,
                                        cells_per_task=args["cells_per_task"],
                                        halo_prop_list=halo_prop_list)
    else:
        cellgrid = None

    # Make sure all ranks have a copy of the cell grid
    cellgrid = comm.bcast(cellgrid)

    # Rank 0 starts a new thread which is going to hand out tasks
    if comm_rank == 0:
        task_queue_thread = threading.Thread(target=distribute_tasks, args=(task_list,))
        task_queue_thread.start()

    # All ranks request and run tasks
    result = []
    while True:
        comm.send(comm_rank, 0, tag=REQUEST_TASK_TAG)
        task = comm.recv(tag=ASSIGN_TASK_TAG) # Must receive from other thread, not self!
        if task is not None:
            result.append(task.run(cellgrid))
        else:
            break

    # Wait for task distributing thread to finish
    if comm_rank == 0:
        task_queue_thread.join()

    # Combine results
    if comm_rank > 0:

        # Ranks >0 send their lists of results to rank 0
        comm.send(result, 0, tag=RETURN_RESULT_TAG)

    else:

        # Rank 0 assembles full result set.
        # First, receive list of results from each other task
        result = []
        for i in range(1, comm_size):
            result += comm.recv(source=i, tag=RETURN_RESULT_TAG)

        # Then combine the results into one array per quantity
        all_results = {}
        for name in result[0].keys():
            list_of_arrays = [r[name][0] for r in result]
            output_array   = np.concatenate(list_of_arrays)
            description    = result[0][name][1]
            all_results[name] = [output_array, description]

        # Sort all arrays by halo index
        idx = np.argsort(all_results["index"][0])
        for name in all_results:
            all_results[name][0] = all_results[name][0][idx,...]

        # And write the output file
        with h5py.File(args["outfile"], "w") as outfile:
            for name, (data, description) in all_results.items():
                outfile[name] = data
                outfile[name].attrs["Description"] = description
                if hasattr(data, "unit"):
                    swift_units.write_unit_attributes(outfile[name], data.unit)
