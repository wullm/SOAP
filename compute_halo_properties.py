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

import halo_centres
import swift_cells
import chunk_tasks
import swift_units
import halo_properties
import task_queue
import lustre
import command_line_args
import SO_properties
import subhalo_properties
import exclusive_sphere_properties
import result_set


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


def sub_snapnum(filename, snapnum):
    """
    Substitute the snapshot number into a filename format string
    without substituting the file number.
    """
    filename = filename.replace("%(file_nr)", "%%(file_nr)")
    return filename % {"snap_nr" : snapnum}


if __name__ == "__main__":

    # Read command line parameters
    args = command_line_args.get_halo_props_args(comm_world)

    # Enable profiling, if requested
    if args.profile == 2 or (args.profile==1 and comm_world_rank==0):
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()

    # Start the clock
    comm_world.barrier()
    t0 = time.time()

    # Split MPI ranks according to which node they are on.
    # Only the first rank on each node belongs to comm_inter_node.
    # Others have comm_inter_node=MPI_COMM_NULL and inter_node_rank=-1.
    comm_intra_node, comm_inter_node = split_comm_world()
    intra_node_rank, intra_node_size = get_rank_and_size(comm_intra_node)
    inter_node_rank, inter_node_size = get_rank_and_size(comm_inter_node)

    # Report number of ranks, compute nodes etc
    if comm_world_rank == 0:
        print("Starting halo properties calculation on %d MPI ranks" % comm_world_size)
        print("Can process %d chunks in parallel using %d ranks per chunk" % (inter_node_size, intra_node_size))

    # Open the snapshot and read SWIFT cell structure, units etc
    if comm_world_rank == 0:
        swift_filename = sub_snapnum(args.swift_filename, args.snapshot_nr)
        extra_input    = sub_snapnum(args.extra_input, args.snapshot_nr)
        if args.reference_snapshot is not None:
            swift_filename_ref = sub_snapnum(args.swift_filename, args.reference_snapshot)
            extra_input_ref    = sub_snapnum(args.extra_input, args.reference_snapshot)
        else:
            swift_filename_ref = None
            extra_input_ref = None
        cellgrid = swift_cells.SWIFTCellGrid(swift_filename, extra_input, swift_filename_ref, extra_input_ref)
        parsec_cgs = cellgrid.constants["parsec"]
        solar_mass_cgs = cellgrid.constants["solar_mass"]
        a = cellgrid.a
    else:
        cellgrid = None
        parsec_cgs = None
        solar_mass_cgs = None
        a = None
    cellgrid, parsec_cgs, solar_mass_cgs, a = comm_world.bcast((cellgrid, parsec_cgs, solar_mass_cgs, a))

    # Get the full list of property calculations we can do
    halo_prop_list = [
        subhalo_properties.SubhaloMasses(cellgrid, bound_only=True),
        subhalo_properties.SubhaloMasses(cellgrid, bound_only=False),
        SO_properties.SOProperties(cellgrid, 50., "mean"),
        SO_properties.SOProperties(cellgrid, 100., "mean"),
        SO_properties.SOProperties(cellgrid, 200., "mean"),
        SO_properties.SOProperties(cellgrid, 500., "mean"),
        SO_properties.SOProperties(cellgrid, 50., "crit"),
        SO_properties.SOProperties(cellgrid, 100., "crit"),
        SO_properties.SOProperties(cellgrid, 200., "crit"),
        SO_properties.SOProperties(cellgrid, 500., "crit"),
        exclusive_sphere_properties.ExclusiveSphereProperties(cellgrid, 10.),
        exclusive_sphere_properties.ExclusiveSphereProperties(cellgrid, 30.),
        exclusive_sphere_properties.ExclusiveSphereProperties(cellgrid, 50.),
        exclusive_sphere_properties.ExclusiveSphereProperties(cellgrid, 100.),
        exclusive_sphere_properties.ExclusiveSphereProperties(cellgrid, 300.),
        exclusive_sphere_properties.ExclusiveSphereProperties(cellgrid, 500.),
        exclusive_sphere_properties.ExclusiveSphereProperties(cellgrid, 1000.),
        exclusive_sphere_properties.ExclusiveSphereProperties(cellgrid, 3000.),
    ]

    # Determine which calculations we're doing this time
    if args.calculations is not None:

        # Check we recognise all the names specified on the command line
        all_names = [hp.name for hp in halo_prop_list]
        for calc in args.calculations:
            if calc not in all_names:
                raise Exception("Don't recognise calculation name: %s" % calc)

        # Filter out calculations which were not selected
        halo_prop_list = [hp for hp in halo_prop_list if hp.name in calc]

    if len(halo_prop_list) < 1:
        raise Exception("Must select at least one halo property calculation!")

    # Report calculations to do
    if comm_world_rank == 0:
        print("Halo property calculations enabled:")
        for hp in halo_prop_list:
            print("  %s" % hp.name)
        if args.centrals_only:
            print("for central halos only")
        else:
            print("for central and satellite halos")

    # Ensure output dir exists
    if comm_world_rank == 0:
        lustre.ensure_output_dir(args.output_file)
    comm_world.barrier()

    # Read in the halo catalogue:
    # All ranks read the file(s) in then gather to rank 0. Also computes search radius for each halo.
    vr_basename = sub_snapnum(args.vr_basename, args.snapshot_nr)
    so_cat = halo_centres.SOCatalogue(comm_world, vr_basename, cellgrid.a_unit,
                                      cellgrid.snap_unit_registry, cellgrid.boxsize,
                                      args.max_halos[0], args.centrals_only,
                                      halo_prop_list)

    # Generate the chunk task list
    if comm_world_rank == 0:
        task_list = chunk_tasks.ChunkTaskList(cellgrid, so_cat,
                                              nr_chunks=args.chunks,
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

    # Execute the chunk tasks
    timings = []
    local_results = task_queue.execute_tasks(tasks, args=(cellgrid, comm_intra_node, inter_node_rank, timings),
                                             comm_all=comm_world, comm_master=comm_inter_node,
                                             comm_workers=comm_intra_node, task_type=chunk_tasks.ChunkTask)

    # The result list has one element per chunk processed. Each element is itself a
    # list of result sets, with one element per iteration that was required.
    # Flatten this so we just have a list of result sets on each MPI rank.
    local_results = [item for sublist in local_results for item in sublist]

    # Make a communicator which only contains tasks which have results
    colour = 0 if len(local_results) > 0 else 1
    comm_have_results = comm_world.Split(colour, comm_world_rank)
    
    # Only tasks with results are involved in writing the output file
    if len(local_results) > 0:
        
        # Combine results on this MPI rank so that we have one array per quantity calculated.
        local_results = result_set.concatenate(local_results)

        # Sort all arrays by halo index
        comm_have_results.barrier()
        t0_sort = time.time()
        local_results.parallel_sort("VR/ID", comm_have_results)
        comm_have_results.barrier()
        t1_sort = time.time()
        if comm_have_results.Get_rank() == 0:
            print("Sorting output arrays took %.2fs" % (t1_sort-t0_sort))

        # Open the output file in collective mode
        comm_have_results.barrier()
        t0_write = time.time()
        output_file = sub_snapnum(args.output_file, args.snapshot_nr)
        outfile = h5py.File(output_file, "w", driver="mpio", comm=comm_have_results)

        # Write metadata copied from snapshot
        cellgrid.write_metadata(outfile.create_group("SWIFT"))

        # Write command line parameters
        params = outfile.create_group("Parameters")
        params.attrs["swift_filename"] = args.swift_filename
        params.attrs["vr_basename"]    = args.vr_basename
        params.attrs["snapshot_nr"]    = args.snapshot_nr
        params.attrs["centrals_only"]  = 0 if args.centrals_only==False else 1
        calc_names = sorted([hp.name for hp in halo_prop_list])
        params.attrs["calculations"] = calc_names

        # Write the halo property arrays
        local_results.collective_write(outfile, comm_have_results)
            
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

    # Save profiling results for each MPI rank
    if args.profile == 2 or (args.profile==1 and comm_world_rank==0):
        pr.disable()
        # Save profile so it can be loaded back into python for analysis
        pr.dump_stats("./profile.%d.dat" % comm_world_rank)
        # Dump text version of the profile
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        with open("./profile.%d.txt" % comm_world_rank, "w") as profile_file:
            profile_file.write(s.getvalue())
    
    if comm_world_rank == 0:
        print("Fraction of time spent calculating halo properties = %.2f" % task_time_fraction)
        print("Total elapsed time: %.1f seconds" % (t1-t0))
        print("Done.")
