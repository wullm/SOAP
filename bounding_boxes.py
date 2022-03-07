#!/bin/env python

import sys

import numpy as np
import swift_cells

def find_bounding_boxes(swift_filename, ptype):

    cellgrid = swift_cells.SWIFTCellGrid(swift_filename)
    nx, ny, nz = cellgrid.cell[ptype].shape

    # Loop over cells
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Read particle positions for this cell
                mask = cellgrid.empty_mask()
                mask[i, j, k] = True
                data = cellgrid.read_masked_cells({ptype : ("Coordinates",)}, mask)
                # Find cell extent
                centre = cellgrid.cell[ptype][i,j,k]["centre"]
                cell_min = centre - 0.5*cellgrid.cell_size.value
                cell_max = centre + 0.5*cellgrid.cell_size.value
                # Find range of particle positions
                pos_min = np.amin(data[ptype]["Coordinates"], axis=0)
                pos_max = np.amax(data[ptype]["Coordinates"], axis=0)

if __name__ == "__main__":
    
    #swift_filename="/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.%(file_nr)d.hdf5"

    swift_filename = sys.argv[1]
    ptype = sys.argv[2]

    find_bounding_boxes(swift_filename, ptype)
