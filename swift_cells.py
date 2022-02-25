#!/bin/env python

import collections
import numpy as np
import h5py
import time
import astropy.cosmology
import astropy.units as u
import swift_units
import task_queue
import shared_array
from mpi4py import MPI

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

        file_name    = None
        infile       = None
        dataset_name = None
        dataset      = None
        
    def open_dataset(self, file_name, dataset_name):

        if file_name != self.file_name:
            self.infile = h5py.File(file_name, "r")
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
        data[self.ptype][self.dataset].full[mem_start:mem_end,...] = infile[dataset_name][file_start:file_end,...]


class SWIFTCellGrid:
    
    def __init__(self, filename):

        self.filename = filename

        # Open the input file
        with h5py.File(filename % {"file_nr":0}, "r") as infile:
            
            # Read cosmology
            cosmo = infile["Cosmology"].attrs
            H0  = 100.0*cosmo["h"][0]
            Om0 = cosmo["Omega_m"][0]
            Omega_b = cosmo["Omega_b"][0]
            Omega_lambda = cosmo["Omega_lambda"][0]
            Omega_r = cosmo["Omega_r"][0]
            Omega_m = cosmo["Omega_m"][0]
            w_0 = cosmo["w_0"][0]
            w_a = cosmo["w_a"][0]
            Tcmb0 = cosmo["T_CMB_0 [K]"][0]
            self.cosmology = astropy.cosmology.w0waCDM(H0=H0, Om0=Omega_m, Ode0=Omega_lambda,
                                                       w0=w_0, wa=w_a, Tcmb0=Tcmb0, Ob0=Omega_b)
            self.a = cosmo["Scale-factor"][0]
            self.z = cosmo["Redshift"][0]
            self.nr_files = infile["Header"].attrs["NumFilesPerSnapshot"][0]
            self.ptypes = [pt for pt in infile["Cells"]["Counts"]]

            # Determine total number of particles of each type in the snapshot
            nr_types = len(infile["Header"].attrs["NumPart_Total"])
            total_nr_particles  =  infile["Header"].attrs["NumPart_Total"].astype(np.int64)
            total_nr_particles += (infile["Header"].attrs["NumPart_Total_HighWord"].astype(np.int64) << 32)
            self.total_nr_particles = {}
            for i in range(nr_types):
                ptype = "PartType%d" % i
                if ptype in self.ptypes:
                    self.total_nr_particles[ptype] = total_nr_particles[i]

            # Read constants
            self.constants = {}
            for name in infile["PhysicalConstants"]["CGS"].attrs:
                self.constants[name] = infile["PhysicalConstants"]["CGS"].attrs[name][0]

            # Read some snapshot units
            self.length_unit = (infile["Units"].attrs["Unit length in cgs (U_L)"][0])*u.cm
            self.mass_unit   = (infile["Units"].attrs["Unit mass in cgs (U_M)"][0])*u.g

            # Get the box size
            self.boxsize = u.Quantity(infile["Header"].attrs["BoxSize"][0], unit=self.length_unit)

            # Read cell meta data
            self.ptypes = []
            self.nr_cells  = infile["Cells/Meta-data"].attrs["nr_cells"]
            self.dimension = infile["Cells/Meta-data"].attrs["dimension"]
            self.cell_size = u.Quantity(infile["Cells/Meta-data"].attrs["size"], unit=self.length_unit)
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
        #
        # Scan files to find shape and dtype for all quantities in the snapshot
        #
        # Make a dict to store the metadata. Aim is to be able to do
        # shape, dtype = self.metadata[ptype][property_name].
        #
        self.metadata = {ptype : {} for ptype in self.ptypes}

        # Make a dict of flags of which particle types we still need to find
        to_find = {ptype : (self.total_nr_particles[ptype] > 0) for ptype in self.ptypes}

        # Loop over files in the snapshot
        for file_nr in range(self.nr_files):
            infile = h5py.File(filename % {"file_nr":file_nr}, "r")
            nr_left = 0
            for ptype in self.ptypes:
                if to_find[ptype]:
                    group_name = ptype
                    if group_name in infile and infile[group_name].attrs["NumberOfParticles"][0] > 0:
                        for name in infile[group_name]:
                            dset = infile[group_name][name]
                            if "a-scale exponent" in dset.attrs:
                                units = swift_units.units_from_attributes(dset)
                                self.metadata[ptype][name] = (dset.shape[1:], dset.dtype, units)
                        to_find[ptype] = False
                    else:
                        nr_left += 1
            infile.close()
            if nr_left == 0:
                break

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
        nr_to_read = len(cells_to_read)
        for cell_nr in range(nr_to_read-1):
            cell1 = cells_to_read[cell_nr]
            cell2 = cells_to_read[cell_nr+1]
            if (cell1["file"] == cell2["file"] and cell1["offset"]+cell1["count"] == cell2["offset"]):
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

    def read_masked_cells(self, property_names, mask):
        """
        Read the requested properties from the cells where mask=True.

        property_names[ptype] contains the list of quantity names to
        read for particle type ptype.

        mask is a 3D boolean array which flags the cells to read.

        Returns a dict with the data arrays:

        data[ptype][property_name] contains property property_name
        for particle type ptype.
        """
        
        # Find ranges of particles to read from each file
        reads = {ptype : self.prepare_read(ptype, mask) for ptype in property_names}

        # Find union of file numbers to read for all particle types
        all_file_nrs = []
        for ptype in property_names:
            all_file_nrs += list(reads[ptype])
        all_file_nrs = np.unique(all_file_nrs)

        # Find number of particles of each type to be read
        nr_parts = {ptype : 0 for ptype in property_names}
        for ptype in property_names:
            for file_nr in reads[ptype]:
                for file_offset, mem_offset, count in reads[ptype][file_nr]:
                    nr_parts[ptype] += count

        # Allocate output arrays
        data = {}
        for ptype in property_names:
            data[ptype] = {}
            for name in property_names[ptype]:
                shape, dtype, units = self.metadata[ptype][name]
                shape = (nr_parts[ptype],)+shape
                data[ptype][name] = u.Quantity(np.ndarray(shape, dtype=dtype), unit=units, dtype=dtype)

        # Loop over files to read
        for file_nr in all_file_nrs:

            # Open this file
            filename = self.filename % {"file_nr" : file_nr}
            infile = h5py.File(filename, "r", rdcc_nbytes=rdcc_nbytes)

            # Loop over particle types to read
            for ptype in property_names:

                # Check if we need to read from this file for this particle type
                if file_nr in reads[ptype]:

                    # Loop over quantities to read for this particle type
                    for name in property_names[ptype]:

                        # Find the dataset
                        dataset = infile[ptype][name]

                        # Read the chunks for this property
                        last_offset = None
                        for (file_offset, mem_offset, count) in reads[ptype][file_nr]:
                            unit = data[ptype][name].unit
                            dtype = data[ptype][name].dtype
                            data[ptype][name][mem_offset:mem_offset+count,...] = (
                                u.Quantity(dataset[file_offset:file_offset+count,...], unit=unit, dtype=dtype))
                            last_offset = file_offset + count
                            mem_offset += count

            # Close the file
            infile.close()

        # Calculate amount of data read
        nbytes = 0
        for ptype in property_names:
            for name in property_names[ptype]:
                nbytes += data[ptype][name].nbytes
        mb_read = nbytes/(1024*1024)

        return data

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

    def read_masked_cells_mpi(self, property_names, mask, comm):
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
        
        # Create read tasks in the required order:
        # By file, then by particle type, then by dataset, then by offset in the file
        all_tasks = collections.deque()
        nr_parts = {ptype : 0 for ptype in property_names}
        for file_nr in all_file_nrs:
            filename = self.filename % {"file_nr" : file_nr}
            for ptype in property_names:
                for dataset in property_names[ptype]:
                    for (file_offset, mem_offset, count) in reads_for_type[ptype][file_nr]:
                        all_tasks.append(ReadTask(filename, ptype, dataset, file_offset, mem_offset, count))
                        nr_parts[ptype] += count

        # Make one task queue per MPI rank
        task_queue = [collections.deque() for _ in range(comm_size)]

        # Share tasks over the task queues roughly equally by number
        nr_tasks = len(all_tasks)
        tasks_per_rank = nr_tasks // comm_size
        for rank in range(comm_size):
            for _ in range(tasks_per_rank):
                task_queue[rank].append(all_tasks.popleft())
            if rank < nr_tasks % comm_size:
                task_queue[rank].append(all_tasks.popleft())
        assert len(all_tasks) == 0

        # Allocate shared storage for the particle data
        data = {}
        for ptype in property_names:
            data[ptype] = {}
            for name in property_names[ptype]:
                shape, dtype, units = self.metadata[ptype][name]
                if comm_rank == 0:
                    shape = (nr_parts[ptype],)+shape
                else:
                    shape = (0,)+shape
                data[ptype][name] = shared_array.SharedArray(shape, dtype, comm, units)
        
        # Execute the tasks
        cache = DatasetCache()
        task_queue.execute_tasks(task_queue, args=(data, cache),
                                 comm_master=comm, comm_workers=MPI.COMM_SELF,
                                 queue_per_rank=True)
        cache.close()

        return data

