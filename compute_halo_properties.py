#!/bin/env python

# Initialize mpi4py with thread support
import mpi4py
mpi4py.rc.threads=True
from mpi4py import MPI
comm_world = MPI.COMM_WORLD
comm_world_rank = comm_world.Get_rank()
comm_world_size = comm_world.Get_size()

import sys
import time
import numpy as np
import h5py
import astropy.units

import halo_centres
import swift_cells
import chunk_tasks
import swift_units
import halo_properties
import task_queue


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


if __name__ == "__main__":

    # Read command line parameters
    args = {}
    if comm_world_rank == 0:
        args["swift_filename"] = sys.argv[1] # Name of one snapshot file
        args["vr_basename"]    = sys.argv[2] # Name of properties file, minus the trailing .N
        args["cells_per_task"] = int(sys.argv[3]) # 1D size of each task in top level cells
        args["outfile"]        = sys.argv[4] # Name of the output file
    args = comm_world.bcast(args)

    # Start the clock
    comm_world.barrier()
    t0 = time.time()
    if comm_world_rank == 0:
        print("Starting halo properties calculation")

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
        task_list = chunk_tasks.ChunkTaskList(cellgrid, so_cat, search_radius=search_radius,
                                              cells_per_task=args["cells_per_task"],
                                              halo_prop_list=halo_prop_list)
        tasks = task_list.tasks
    else:
        cellgrid = None
        tasks = None

    # All ranks will need the cell grid
    cellgrid = comm_world.bcast(cellgrid)

    # Periodic boundary is only implemented for tasks smaller than the full box
    for ptype in cellgrid.ptypes:
        for i in range(3):
            if args["cells_per_task"] > cellgrid.cell[ptype].shape[i]/2:
                raise Exception("cells_per_task is too large!")

    # Split MPI ranks according to which node they are on.
    # Only the first rank on each node belongs to comm_inter_node.
    # Others have comm_inter_node=MPI_COMM_NULL and inter_node_rank=-1.
    comm_intra_node, comm_inter_node = split_comm_world()
    intra_node_rank, intra_node_size = get_rank_and_size(comm_intra_node)
    inter_node_rank, inter_node_size = get_rank_and_size(comm_inter_node)

    # Execute the tasks
    result = task_queue.execute_tasks(tasks, args=(cellgrid, comm_intra_node, inter_node_rank),
                                      comm_all=comm_world, comm_master=comm_inter_node,
                                      comm_workers=comm_intra_node)

    # Combine results
    if comm_world_rank > 0:

        # Ranks>0 send their lists of results to rank 0
        comm_world.send(result, 0)

    elif comm_world_rank == 0:

        # Rank 0 assembles full result set.
        # First, receive list of results from each other task
        result = []
        for i in range(1, comm_world_size):
            result += comm_world.recv(source=i)

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

    # Stop the clock
    comm_world.barrier()
    t1 = time.time()
    if comm_world_rank == 0:
        print("Total elapsed time: %.1f seconds" % (t1-t0))
        print("Done.")
