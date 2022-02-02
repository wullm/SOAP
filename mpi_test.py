#!/bin/env python3

import sys

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

if __name__ == "__main__":

    # Read command line parameters
    args = {}
    if comm_rank == 0:
        args["swift_filename"] = sys.argv[1] # Name of one snapshot file
        args["vr_basename"]    = sys.argv[2] # Name of properties file, minus the trailing .N
        args["cells_per_task"] = int(sys.argv[3]) # 1D size of each task in top level cells
        args["outfile"]        = sys.argv[4] # Name of the output file
    args = comm.bcast(args)

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
                                        cells_per_task=args["cells_per_task"])
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
                print("Starting task %d" % next_task)
                comm.send(task_list.tasks[next_task], request_src)
                next_task += 1
            else:
                comm.send(None, request_src)
                nr_done += 1
                print("Number of ranks done with all tasks = %d" % nr_done)
        print("All tasks done.")

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

    # Barrier prevents mix-up between task requests and combining results
    # (could fix with tags)
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
        with h5py.File(args["outfile"], "w") as outfile:
            for name in all_results:
                outfile[name] = all_results[name]
                if hasattr(all_results[name], "unit"):
                    swift_units.write_unit_attributes(outfile[name], all_results[name].unit)
