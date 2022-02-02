#!/bin/env python3

import sys

import numpy as np
import h5py
import astropy.units

import halo_centres
import swift_cells
import so_tasks
import swift_units


if __name__ == "__main__":

    # Read command line parameters
    args = {}
    args["swift_filename"] = sys.argv[1] # Name of one snapshot file
    args["vr_basename"]    = sys.argv[2] # Name of properties file, minus the trailing .N
    args["cells_per_task"] = int(sys.argv[3]) # 1D size of each task in top level cells
    args["outfile"]        = sys.argv[4] # Name of the output file

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

    # Run tasks
    result = []
    #for i, task in enumerate(task_list.tasks):
    #    print("Run task %d" % i)
    #    result.append(task.run(cellgrid))

    result.append(task_list.tasks[7].run(cellgrid))

    # Combine results from tasks
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
