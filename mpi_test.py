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

comm_world = MPI.COMM_WORLD
comm_world_rank = comm_world.Get_rank()

# Message tags required to prevent mix-ups!
REQUEST_TASK_TAG=1
ASSIGN_TASK_TAG=2
RETURN_RESULT_TAG=3


def split_comm_world():

    # Communicator containing all ranks on this node
    comm_intra_node = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
    comm_intra_node_rank = comm_intra_node.Get_rank()

    # Communicator containing first rank on each node only:
    # other ranks will have comm_inter_node=MPI_COMM_NULL.
    colour = 0 if comm_intra_node_rank==0 else MPI.UNDEFINED
    key = MPI.COMM_WORLD.Get_rank()
    comm_inter_node = MPI.COMM_WORLD.Split(colour, key)

    return comm_intra_node, comm_inter_node


def get_rank_and_size(comm):
    if comm == MPI.COMM_NULL:
        return (-1, -1)
    else:
        return (comm.Get_rank(), comm.Get_size())


def sleepy_recv(comm, tag):
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


def distribute_tasks(task_list, comm):
    """
    Listen for and respond to requests for tasks to do
    """
    comm_size = comm.Get_size()
    next_task = 0
    nr_tasks = len(task_list.tasks)
    nr_done = 0
    while nr_done < comm_size:
        request_src = sleepy_recv(comm, REQUEST_TASK_TAG)
        if next_task < nr_tasks:
            print("Starting task %d of %d on node %d" % (next_task, nr_tasks, request_src))
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
    if comm_world_rank == 0:
        args["swift_filename"] = sys.argv[1] # Name of one snapshot file
        args["vr_basename"]    = sys.argv[2] # Name of properties file, minus the trailing .N
        args["cells_per_task"] = int(sys.argv[3]) # 1D size of each task in top level cells
        args["outfile"]        = sys.argv[4] # Name of the output file
    args = comm_world.bcast(args)

    # Make a list of properties to calculate
    halo_prop_list = [halo_properties.SOMasses(),]

    # Rank zero reads the halo positions and generates a list of tasks
    if comm_world_rank == 0:

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
    cellgrid = comm_world.bcast(cellgrid)

    # Split MPI ranks according to which node they are on.
    # Only the first rank on each node belongs to comm_inter_node.
    # Others have comm_inter_node=MPI_COMM_NULL and inter_node_rank=-1.
    comm_intra_node, comm_inter_node = split_comm_world()
    intra_node_rank, intra_node_size = get_rank_and_size(comm_intra_node)
    inter_node_rank, inter_node_size = get_rank_and_size(comm_inter_node)

    # Node 0's first rank starts a new thread which is going to hand out tasks to each node
    if inter_node_rank == 0:
        task_queue_thread = threading.Thread(target=distribute_tasks, args=(task_list, comm_inter_node))
        task_queue_thread.start()

    # Request and run tasks until there are none left
    result = []
    while True:

        # The first rank on the node requests a task and broadcasts it to all ranks on the node
        if intra_node_rank == 0:
            comm_inter_node.send(inter_node_rank, 0, tag=REQUEST_TASK_TAG)
            task = comm_inter_node.recv(tag=ASSIGN_TASK_TAG)
        else:
            task = None
        task = comm_intra_node.bcast(task)

        # All ranks on this node execute the task as a collective operation
        if task is not None:
            result.append(task.run(cellgrid, comm_intra_node)) # Result only returned by intra_node_rank==0?
        else:
            break

    # Wait for task distributing thread to finish
    if inter_node_rank == 0:
        task_queue_thread.join()

    # Combine results
    if inter_node_rank > 0:

        # Nodes>0 send their lists of results to node 0
        comm_inter_node.send(result, 0, tag=RETURN_RESULT_TAG)

    elif inter_node_rank == 0:

        # Node 0 assembles full result set.
        # First, receive list of results from each other task
        result = []
        for i in range(1, inter_node_size):
            result += comm_inter_node.recv(source=i, tag=RETURN_RESULT_TAG)

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
