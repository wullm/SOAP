#!/bin/env python

from mpi4py import MPI 
import numpy as np 
import astropy.units as u


class SharedArray:

    def __init__(self, local_shape, dtype, comm, units=None):
        
        self.comm = comm
        self.dtype = np.dtype(dtype)

        # Find the full size of the array
        n = comm.allreduce(local_shape[0])
        full_shape = (n,)+local_shape[1:]

        # Find amount of memory to allocate on this rank
        local_elements = 1
        for s in local_shape:
            local_elements *= s

        # Find amount of memory to allocate in total
        full_elements = 1
        for s in full_shape:
            full_elements *= s

        # Allocate shared memory window
        self.win = MPI.Win.Allocate_shared(local_elements*self.dtype.itemsize,
                                           self.dtype.itemsize, comm=comm) 

        # Make a numpy array to access the full array
        buf, itemsize = self.win.Shared_query(0)
        nbytes_all = full_elements*itemsize
        buf = MPI.memory.fromaddress(buf.address, nbytes_all)
        self.full = np.ndarray(buffer=buf, dtype=self.dtype, shape=full_shape)

        # Make a numpy array to access the local part of the array
        buf, itemsize = self.win.Shared_query(comm.Get_rank())
        self.local = np.ndarray(buffer=buf, dtype=self.dtype, shape=local_shape)

        # Add units if specified
        if units is not None:
            self.full  = u.Quantity(self.full, unit=units, copy=False)
            self.local = u.Quantity(self.local, unit=units, copy=False)

    def sync(self):
        self.win.Sync()
        self.comm.barrier()
        self.win.Sync()

    def free(self):
        if self.win is not None:
            self.win.Free()
            self.win = None


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    
    local_shape = (10,)
    arr = SharedArray(local_shape, np.float64, comm)
    arr.sync()

    # Rank 0 writes elements
    if comm_rank == 0:
        arr.full[:] = 27

    arr.sync()
    
    # Rank 1 reads
    if comm_rank == 1:
        print("Minimum value = ", np.amin(arr.full))
        print("Maximum value = ", np.amax(arr.full))

    arr.sync()
    arr.free()

