#!/bin/env python

import time
import matplotlib.pyplot as plt
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import swift_cells


def io_test():

    comm.barrier()
    t0 = time.time()

    # Open the snapshot
    fname = "/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0030/flamingo_0030.%(file_nr)d.hdf5"
    cellgrid = swift_cells.SWIFTCellGrid(fname)
    
    # Quantities to read
    property_names = {
        "PartType0" : ("Coordinates", "Masses"),
        "PartType1" : ("Coordinates", "Masses"),
    }

    # Specify region to read
    pos_min = np.asarray((    0.0,    0.0,    0.0))
    pos_max = np.asarray((  500.0,  500.0,  500.0))

    # Read in the region
    mask = cellgrid.empty_mask()
    cellgrid.mask_region(mask, pos_min, pos_max)
    data = cellgrid.read_masked_cells_to_shared_memory(property_names, mask, comm)
    #data = cellgrid.read_masked_cells(property_names, mask)

    comm.barrier()
    t1 = time.time()

    # Find read rate
    nbytes = 0
    for ptype in property_names:
        for dataset in property_names[ptype]:
            arr = data[ptype][dataset].full
            nbytes += arr.data.nbytes
    elapsed = t1 - t0

    if comm_rank == 0:
        rate = nbytes/elapsed/(1024**3)
        print("Read at %.2f GB/sec on %d ranks" % (rate, comm_size))

    # # Plot the particles
    # if comm_rank == 0:
    #     pos = data["PartType1"]["Coordinates"].full
    #     #pos = data["PartType1"]["Coordinates"]
    #     plt.plot(pos[:,0], pos[:,1], "k,", alpha=0.2)
    #     plt.gca().set_aspect("equal")
    #     plt.show()


if __name__ == "__main__":
    io_test()
