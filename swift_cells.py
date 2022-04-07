#!/bin/env python

import collections

import numpy as np
import h5py
import time
from mpi4py import MPI
import swiftsimio

import swift_units
import task_queue
import shared_array

# HDF5 chunk cache parameters:
# SWIFT writes datasets with large chunks so the default 1Mb may be too small
# (e.g. one uncompressed chunk of positions is ~24Mb in FLAMINGO 2.8Gpc run)
rdcc_nbytes = 250*1024*1024

# Type to store information about a SWIFT cell for one particle type
swift_cell_t = np.dtype([
    ("centre", np.float64, 3), # coordinates of cell centre
    ("count",  np.int64),      # number of particles in the cell
    ("offset", np.int64),      # offset to first particle
    ("file",   np.int32),      # file containing this cell
    ("order",  np.int32),      # ordering of the cells in the snapshot file(s)
])


class DatasetCache:
    """
    Class to allow h5py File and Dataset objects to persist
    between ReadTask invocations.
    """
    def __init__(self):

        self.file_name    = None
        self.infile       = None
        self.dataset_name = None
        self.dataset      = None
        
    def open_dataset(self, file_name, dataset_name):

        if file_name != self.file_name:
            self.infile = h5py.File(file_name, "r", rdcc_nbytes=rdcc_nbytes)
            self.file_name = file_name
            self.dataset_name = None
            self.dataset = None

        if dataset_name != self.dataset_name:
            self.dataset = self.infile[dataset_name]
            self.dataset_name = dataset_name
            
        return self.dataset

    def close(self):
        self.dataset = None
        self.dataset_name = None
        if self.infile is not None:
            self.infile.close()
        self.file_name = None

class ReadTask:
    """
    Class to execute a read of a single contiguous chunk of an array
    """
    def __init__(self, file_name, ptype, dataset, file_offset, mem_offset, count):

        self.file_name   = file_name
        self.ptype       = ptype
        self.dataset     = dataset
        self.file_offset = file_offset
        self.mem_offset  = mem_offset
        self.count       = count

    def __call__(self, data, cache):
     
        # Find the dataset
        dataset_name = self.ptype+"/"+self.dataset
        dataset = cache.open_dataset(self.file_name, dataset_name)

        # Read the data
        mem_start = self.mem_offset
        mem_end   = self.mem_offset + self.count
        file_start = self.file_offset
        file_end   = self.file_offset + self.count

        dataset.read_direct(data[self.ptype][self.dataset].full,
                            np.s_[file_start:file_end,...],
                            np.s_[mem_start:mem_end,...])


def identify_datasets(filename, nr_files, ptypes, unit_system, a):
    """
    Find units, data type and shape for datasets in snapshot-like files.
    Returns a dict with one entry per particle type. Dict keys are the
    property names and values are (shape, dtype, units) tuples.
    """
    metadata = {ptype : {} for ptype in ptypes}

    # Make a dict of flags of which particle types we still need to find
    to_find = {ptype : True for ptype in ptypes}

    # Scan snapshot files to find shape, type and units for each quantity
    for file_nr in range(nr_files):
        infile = h5py.File(filename % {"file_nr":file_nr}, "r")
        nr_left = 0
        for ptype in ptypes:
            if to_find[ptype]:
                group_name = ptype
                if group_name in infile:
                    for name in infile[group_name]:
                        dset = infile[group_name][name]
                        if "a-scale exponent" in dset.attrs:
                            cosmo = swift_units.empty_cosmo_array_from_attributes(dset, unit_system, a)
                            metadata[ptype][name] = cosmo
                    to_find[ptype] = False
                else:
                    nr_left += 1
        infile.close()
        if nr_left == 0:
            break

    return metadata


class SWIFTCellGrid:
    
    def __init__(self, snap_filename, extra_filename=None):

        self.snap_filename = snap_filename

        # Option format string to generate name of file(s) with extra datasets
        self.extra_filename = extra_filename

        # Use swiftsimio to get the cosmology, units etc from the snapshot
        snap = swiftsimio.load(snap_filename % {"file_nr":0})
        self.cosmology = snap.metadata.cosmology
        self.units = snap.metadata.units
        self.boxsize = snap.metadata.boxsize[0]
        self.a = snap.metadata.a
        self.z = snap.metadata.z
        self.nr_files = snap.metadata.header["NumFilesPerSnapshot"][0]

        # Determine which particle types are present
        self.ptypes = ["PartType%d" % i for i in snap.metadata.present_particle_types]
        
        # Open the input file
        with h5py.File(snap_filename % {"file_nr":0}, "r") as infile:
            
            # Read constants
            self.constants = {}
            for name in infile["PhysicalConstants"]["CGS"].attrs:
                self.constants[name] = infile["PhysicalConstants"]["CGS"].attrs[name][0]

            # Read cell meta data
            self.ptypes = []
            self.nr_cells  = infile["Cells/Meta-data"].attrs["nr_cells"]
            self.dimension = infile["Cells/Meta-data"].attrs["dimension"]
            self.cell_size = infile["Cells/Meta-data"].attrs["size"]*self.units.length
            for name in infile["Cells/Counts"]:
                self.ptypes.append(name)
                
            # Create arrays of cells
            self.cell = {}
            for ptype in self.ptypes:
                self.cell[ptype] = np.ndarray(self.nr_cells, dtype=swift_cell_t)
                
            # Read cell info
            for ptype in self.ptypes:
                cellgrid = self.cell[ptype]
                cellgrid["centre"] = infile["Cells/Centres"][...]
                cellgrid["count"]  = infile["Cells"]["Counts"][ptype][...]
                cellgrid["offset"] = infile["Cells"]["OffsetsInFile"][ptype][...]
                cellgrid["file"]   = infile["Cells"]["Files"][ptype][...]

        # Determine ordering of the cells in the snapshot
        for ptype in self.ptypes:
            cellgrid = self.cell[ptype]
            idx = np.lexsort((cellgrid["offset"], cellgrid["file"]))
            for cell_order, cell_index in enumerate(idx):
                cellgrid[cell_index]["order"] = cell_order

        # Reshape into a grid
        for ptype in self.ptypes:
            self.cell[ptype] = self.cell[ptype].reshape(self.dimension)

        # Scan files to find shape and dtype etc for all quantities in the snapshot.
        self.snap_metadata = identify_datasets(snap_filename, self.nr_files, self.ptypes, self.units, self.a)
        if extra_filename is not None:
            self.extra_metadata = identify_datasets(extra_filename, self.nr_files, self.ptypes, self.units, self.a)

    def prepare_read(self, ptype, mask):
        """
        Determine which ranges of particles we need to read from each file
        to read all of the cells indicated by the mask for the specified
        particle type.

        ptype - which particle type to read
        mask  - 3D boolean array with one element per cell, true if the
                cell is to be read and false otherwise

        Returns a dict where the keys are the unique file numbers to read
        and the values are lists of (offset_in_file, offset_in_memory, count) tuples.
        """
        
        # Make an array of the selected cells
        cells_to_read = self.cell[ptype][mask].flatten()
        
        # Discard any empty cells
        cells_to_read = cells_to_read[cells_to_read["count"] > 0]

        # Sort the selected cells by file, and then by offset within the file
        idx = np.lexsort((cells_to_read["offset"], cells_to_read["file"]))
        cells_to_read = cells_to_read[idx]

        # Merge adjacent cells
        max_size = 20*1024**2
        nr_to_read = len(cells_to_read)
        for cell_nr in range(nr_to_read-1):
            cell1 = cells_to_read[cell_nr]
            cell2 = cells_to_read[cell_nr+1]
            if (cell1["file"] == cell2["file"] and cell1["offset"]+cell1["count"] == cell2["offset"] and (cell1["count"]+cell2["count"]) <= max_size):
                # Merge cells: put the particles in cell2 and empty cell1
                cell2["count"] += cell1["count"]
                cell2["offset"] = cell1["offset"]
                cell1["count"] = 0

        # Discard any cells which are now empty due to the merging
        cells_to_read = cells_to_read[cells_to_read["count"] > 0]

        # Find unique file numbers to read
        unique_file_nrs = np.unique(cells_to_read["file"])

        # Make a list of reads for each file:
        # reads[file_nr] is a list of (file_offset, memory_offset, count) tuples for file file_nr.
        mem_offset = 0
        reads = {file_nr : [] for file_nr in unique_file_nrs}
        for cell in cells_to_read:
            reads[cell["file"]].append((cell["offset"], mem_offset, cell["count"]))
            mem_offset += cell["count"]

        return reads

    def empty_mask(self):

        return np.zeros(self.dimension, dtype=np.bool)

    def mask_region(self, mask, pos_min, pos_max):
        imin = np.asarray(np.floor(pos_min/self.cell_size), dtype=int)
        imax = np.asarray(np.floor(pos_max/self.cell_size), dtype=int)
        for i in range(imin[0], imax[0]+1):
            ii = i % self.dimension[0]
            for j in range(imin[1], imax[1]+1):
                jj = j % self.dimension[1]
                for k in range(imin[2], imax[2]+1):
                    kk = k % self.dimension[2]
                    mask[ii,jj,kk] = True

    def read_masked_cells_to_shared_memory(self, property_names, mask, comm):
        """
        Read in the specified properties for the cells with mask=True
        """
        
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

        # Make a list of all reads to execute for each particle type
        reads_for_type = {}
        for ptype in property_names:
            reads_for_type[ptype] = self.prepare_read(ptype, mask)
            
        # Find union of file numbers to read for all particle types
        all_file_nrs = []
        for ptype in property_names:
            all_file_nrs += list(reads_for_type[ptype])
        all_file_nrs = np.unique(all_file_nrs)
        
        # Count particles to read in
        nr_parts = {ptype : 0 for ptype in property_names}
        for file_nr in all_file_nrs:
            for ptype in property_names:
                for (file_offset, mem_offset, count) in reads_for_type[ptype][file_nr]:
                    nr_parts[ptype] += count

        # Create read tasks in the required order:
        # By file, then by particle type, then by dataset, then by offset in the file
        all_tasks = collections.deque()
        for file_nr in all_file_nrs:
            filename = self.snap_filename % {"file_nr" : file_nr}
            for ptype in property_names:
                for dataset in property_names[ptype]:
                    if dataset in self.snap_metadata[ptype]:
                        for (file_offset, mem_offset, count) in reads_for_type[ptype][file_nr]:
                            all_tasks.append(ReadTask(filename, ptype, dataset, file_offset, mem_offset, count))

        # Create additional read tasks for the extra data files
        if self.extra_filename is not None:
            for file_nr in all_file_nrs:
                filename = self.extra_filename % {"file_nr" : file_nr}
                for ptype in property_names:
                    for dataset in property_names[ptype]:
                        if dataset in self.extra_metadata[ptype]:
                            for (file_offset, mem_offset, count) in reads_for_type[ptype][file_nr]:
                                all_tasks.append(ReadTask(filename, ptype, dataset, file_offset, mem_offset, count))

        # Make one task queue per MPI rank
        tasks = [collections.deque() for _ in range(comm_size)]

        # Share tasks over the task queues roughly equally by number
        nr_tasks = len(all_tasks)
        tasks_per_rank = nr_tasks // comm_size
        for rank in range(comm_size):
            for _ in range(tasks_per_rank):
                tasks[rank].append(all_tasks.popleft())
            if rank < nr_tasks % comm_size:
                tasks[rank].append(all_tasks.popleft())
        assert len(all_tasks) == 0

        # Allocate MPI shared memory for the particle data
        data = {}
        for ptype in property_names:
            data[ptype] = {}
            for name in property_names[ptype]:

                # Get metadata for array to allocate in memory
                if name in self.snap_metadata[ptype]:
                    arr = self.snap_metadata[ptype][name]
                elif self.extra_metadata is not None and name in self.extra_metadata[ptype]:
                    arr = self.extra_metadata[ptype][name]
                else:
                    raise Exception("Can't find required dataset %s in input file(s)!" % name)
                shape = arr.shape[1:]
                dtype = arr.dtype.newbyteorder("=") # Must be native endian for mpi4py
                units = arr.units
                cosmo_array_params=(units, arr.cosmo_factor, arr.comoving)

                # Determine size of local array section
                nr_local = nr_parts[ptype] // comm_size
                if comm_rank < (nr_parts[ptype] % comm_size):
                    nr_local += 1
                # Find global and local shape of the array
                global_shape = (nr_parts[ptype],)+shape
                local_shape  = (nr_local,)+shape
                # Allocate storage
                data[ptype][name] = shared_array.SharedArray(local_shape, dtype, comm, cosmo_array_params)

        comm.barrier()

        # Execute the tasks
        cache = DatasetCache()
        task_queue.execute_tasks(tasks, args=(data, cache), comm_all=comm, comm_master=comm,
                                 comm_workers=MPI.COMM_SELF, queue_per_rank=True)
        cache.close()
        
        # Ensure all arrays have been fully written
        comm.barrier()
        for ptype in property_names:
            for name in property_names[ptype]:
                data[ptype][name].sync()
        comm.barrier()

        return data

