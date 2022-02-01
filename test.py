#!/bin/env python3

import halo_centres
import swift_cells
import so_tasks

# Location of the input
vr_basename = "/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/vr/catalogue_0013/vr_catalogue_0013.properties"
swift_filename = "/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/snapshots/flamingo_0013.hdf5"

# Read the halo catalogue. TODO: convert to comoving?
so_cat = halo_centres.SOCatalogue(vr_basename)

# Read SWIFT cells
cellgrid = swift_cells.SWIFTCellGrid(swift_filename)

# Generate task list
task_list = so_tasks.SOTaskList(cellgrid, so_cat, cells_per_task=3)

# Decide on search radius (snapshot length units)
max_halo_radius = 10.0
search_radius = max_halo_radius + 0.5*np.amax(cellgrid.cell_size)
