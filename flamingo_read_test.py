#!/bin/env python3

import halo_centres
import swift_cells

# Location of the input
swift_filename="/cosma8/data/dp004/jlvc76/FLAMINGO/ScienceRuns/L2800N5040/HYDRO_FIDUCIAL/snapshots/flamingo_0078/flamingo_0078.%(file_nr)d.hdf5"

# Read SWIFT cells
cellgrid = swift_cells.SWIFTCellGrid(swift_filename)

# Decide on quantities to read
property_names = {
    "PartType1" : ["Coordinates", "Masses", "ParticleIDs"],
}

# Determine which cells to read
pos_min=(1500,2500,1000)
pos_max=(1600,2600,1100)
mask = cellgrid.empty_mask()
cellgrid.mask_region(mask, pos_min, pos_max)

# Read the cells
data = cellgrid.read_masked_cells(property_names, mask, verbose=True)
