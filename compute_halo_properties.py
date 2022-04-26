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
import unyt

import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort

import halo_centres
import swift_cells
import chunk_tasks
import swift_units
import halo_properties
import task_queue
import lustre


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
        args["chunks_per_dimension"] = int(sys.argv[3]) # Number of chunks to divide volume into
        args["outfile"]        = sys.argv[4] # Name of the output file
        if len(sys.argv) > 5:
            args["extra_filename"] = sys.argv[5] # Additional data file, if any
        else:
            args["extra_filename"] = None            
    args = comm_world.bcast(args)

    # Start the clock
    comm_world.barrier()
    t0 = time.time()
    if comm_world_rank == 0:
        print("Starting halo properties calculation on %d MPI ranks" % comm_world_size)

    # Make a list of properties to calculate
    halo_prop_list = [
        halo_properties.SOMasses(),
        halo_properties.CentreOfMass(),
    ]

    # Open the snapshot and read SWIFT cell structure, units etc
    if comm_world_rank == 0:
        cellgrid = swift_cells.SWIFTCellGrid(args["swift_filename"], args["extra_filename"])
        parsec_cgs = cellgrid.constants["parsec"]
        solar_mass_cgs = cellgrid.constants["solar_mass"]
        a = cellgrid.a
    else:
        cellgrid = None
        parsec_cgs = None
        solar_mass_cgs = None
        a = None
    cellgrid, parsec_cgs, solar_mass_cgs, a = comm_world.bcast((cellgrid, parsec_cgs, solar_mass_cgs, a))

    # Ensure output dir exists
    if comm_world_rank == 0:
        lustre.ensure_output_dir(args["outfile"])
    comm_world.barrier()

    # Read in the halo catalogue:
    # All ranks read the file(s) in then gather to rank 0. Also computes search radius for each halo.
    so_cat = halo_centres.SOCatalogue(comm_world, args["vr_basename"], cellgrid.a_unit,
                                      cellgrid.snap_unit_registry, cellgrid.boxsize)

    # Generate the chunk task list
    if comm_world_rank == 0:
        task_list = chunk_tasks.ChunkTaskList(cellgrid, so_cat,
                                              chunks_per_dimension=args["chunks_per_dimension"],
                                              halo_prop_list=halo_prop_list)
        tasks = task_list.tasks
    else:
        tasks = None

    # Report initial set-up time
    comm_world.barrier()
    t1 = time.time()
    if comm_world_rank == 0:
        print("Reading %d VR halos and setting up %d chunk(s) took %.1fs" % (so_cat.nr_halos, len(tasks), t1-t0))

    # We no longer need the VR catalogue, since halo centres etc are stored in the chunk tasks
    del so_cat

    # Split MPI ranks according to which node they are on.
    # Only the first rank on each node belongs to comm_inter_node.
    # Others have comm_inter_node=MPI_COMM_NULL and inter_node_rank=-1.
    comm_intra_node, comm_inter_node = split_comm_world()
    intra_node_rank, intra_node_size = get_rank_and_size(comm_intra_node)
    inter_node_rank, inter_node_size = get_rank_and_size(comm_inter_node)

    # Execute the chunk tasks
    timings = []
    result = task_queue.execute_tasks(tasks, args=(cellgrid, comm_intra_node, inter_node_rank, timings),
                                      comm_all=comm_world, comm_master=comm_inter_node,
                                      comm_workers=comm_intra_node, task_type=chunk_tasks.ChunkTask)

    # Make a communicator which only contains tasks which have results
    colour = 0 if len(result) > 0 else 1
    comm_have_results = comm_world.Split(colour, comm_world_rank)
    
    # Only tasks with results are involved in writing the output file
    if len(result) > 0:

        # Combine results on this MPI rank so that we have one array per quantity calculated.
        local_results = {}
        for name in result[0].keys():
            list_of_arrays = [r[name][0] for r in result]
            output_array = unyt.array.uconcatenate(list_of_arrays, axis=0)
            description    = result[0][name][1]
            local_results[name] = [output_array, description]

        # Get the full list of property names, in consistent order between ranks
        if comm_have_results.Get_rank() == 0:
            names = list(local_results.keys())
        else:
            names = None
        names = comm_have_results.bcast(names)

        # Sort all arrays by halo index
        comm_have_results.barrier()
        t0_sort = time.time()

        idx = psort.parallel_sort(local_results["index"][0], comm=comm_have_results, return_index=True)
        for name in names:
            if name != "index":
                local_results[name][0] = psort.fetch_elements(local_results[name][0], idx, comm=comm_have_results)
        del idx
        comm_have_results.barrier()
        t1_sort = time.time()
        if comm_have_results.Get_rank() == 0:
            print("Sorting output arrays took %.2fs" % (t1_sort-t0_sort))

        # Open the output file in collective mode
        comm_have_results.barrier()
        t0_write = time.time()
        outfile = h5py.File(args["outfile"], "w", driver="mpio", comm=comm_have_results)

        # Loop over output quantities
        for name in names:
            data, description = local_results[name]
            phdf5.collective_write(outfile, name, data, comm_have_results)
            if hasattr(data, "units"):
                attrs = swift_units.attributes_from_units(data.units)
                for attr_name, attr_value in attrs.items():
                    outfile[name].attrs[attr_name] = attr_value
            outfile[name].attrs["Description"] = description
            
        # Finished writing the output
        outfile.close()
        comm_have_results.barrier()
        t1_write = time.time()
        if comm_have_results.Get_rank() == 0:
            print("Writing output took %.2fs" % (t1_write-t0_write))

    # Stop the clock
    comm_world.barrier()
    t1 = time.time()

    # Find total time spent running tasks
    if len(timings) > 0:
        task_time_local = sum(timings)
    else:
        task_time_local = 0.0
    task_time_total = comm_have_results.allreduce(task_time_local)
    task_time_fraction = task_time_total / (comm_world_size*(t1-t0))
    
    if comm_world_rank == 0:
        print("Fraction of time spent calculating halo properties = %.2f" % task_time_fraction)
        print("Total elapsed time: %.1f seconds" % (t1-t0))
        print("Done.")
