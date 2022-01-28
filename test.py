#!/bin/env python3

import halo_centres
import swift_cells

vr_basename = "/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/vr/catalogue_0013/vr_catalogue_0013.properties"

so_cat = halo_centres.SOCatalogue(vr_basename)

swift_filename = "/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/snapshots/flamingo_0013.hdf5"

cellgrid = swift_cells.SWIFTCellGrid(swift_filename)
