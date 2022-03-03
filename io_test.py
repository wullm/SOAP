#!/bin/env python

from mpi4py import MPI
comm = MPI.COMM_WORLD

import swift_cells


def io_test():

    # Open the snapshot
    fname = "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.0.hdf5"
    cellgrid = swift_cells.SWIFTCellGrid(fname)
    
    # Quantities to read
    property_names = {
        "PartType0" : ("Coordinates", "Masses"),
        "PartType1" : ("Coordinates", "Masses"),
    }

    # Specify region to read
    pos_min = (  0.0,   0.0,   0.0)
    pos_max = ( 50.0,  50.0,  50.0)

    # Read in the region
    mask = cellgrid.empty_mask()
    cellgrid.mask_region(mask, pos_min, pos_max)
    data = cellgrid.read_masked_cells_to_shared_memory(property_names, mask, comm)


if __name__ == "__main__":
    io_test()
