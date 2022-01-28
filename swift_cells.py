#!/bin/env python

import numpy as np
import h5py
import time

# HDF5 chunk cache parameters:
# SWIFT writes datasets with large chunks so the default 1Mb may be too small
# (e.g. one uncompressed chunk of positions is ~24Mb in FLAMINGO 2.8Gpc run)
rdcc_nbytes = 250*1024*1024

# Type to store information about a SWIFT cell for one particle type
swift_cell_t = np.dtype([
    ("centre", np.float64, 3),
    ("count",  np.int64),
    ("offset", np.int64),
    ("file",   np.int32),
])

class SWIFTCellGrid:
    
    def __init__(self, filename):

        self.filename = filename

        # Open the input file
        with h5py.File(filename % {"file_nr":0}, "r") as infile:
            
            # Read cell meta data
            self.ptypes = []
            self.nr_cells  = infile["Cells/Meta-data"].attrs["nr_cells"]
            self.dimension = infile["Cells/Meta-data"].attrs["dimension"]
            self.cell_size = infile["Cells/Meta-data"].attrs["size"]
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

        # Reshape into a grid
        for ptype in self.ptypes:
            self.cell[ptype] = self.cell[ptype].reshape(self.dimension)

    def prepare_read(self, ptype, mask):
        """
        Determine which ranges of particles we need to read from each file
        to read all of the cells indicated by the mask for the specified
        particle type.

        ptype - which particle type to read
        mask  - 3D boolean array with one element per cell, true if the
                cell is to be read and false otherwise

        Returns a dict where the keys are the unique file numbers to read
        and the values are lists of (offset, count) tuples.
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
        # reads[file_nr] is a list of (offset, count) tuples for file file_nr.
        reads = {file_nr : [] for file_nr in unique_file_nrs}
        for cell in cells_to_read:
            reads[cell["file"]].append((cell["offset"], cell["count"]))

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

        start = time.time()

        # Dict to store the result. Initially None for each quantity to read.
        data = {}
        for ptype in property_names:
            data[ptype] = {name : None for name in property_names[ptype]}
        
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
                for offset, count in reads[ptype][file_nr]:
                    nr_parts[ptype] += count

        # Will need to store offset into output arrays for each type
        ptype_offset = {ptype : 0 for ptype in property_names}

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

                        # Allocate the output array if necessary
                        if data[ptype][name] is None:
                            dtype = dataset.dtype
                            shape = list(dataset.shape)
                            shape[0] = nr_parts[ptype]
                            data[ptype][name] = np.ndarray(shape, dtype=dtype)

                        # Read the chunks for this property
                        mem_offset = ptype_offset[ptype]
                        for (file_offset, count) in reads[ptype][file_nr]:
                            source_sel = np.s_[file_offset:file_offset+count,...]
                            dest_sel   = np.s_[mem_offset:mem_offset+count,...]
                            dataset.read_direct(data[ptype][name], source_sel, dest_sel)
                            mem_offset += count

                    # Increment offsets into output arrays by number of particles read from this file
                    for (file_offset, count) in reads[ptype][file_nr]:
                        ptype_offset[ptype] += count

            # Close the file
            infile.close()

        # Calculate amount of data read
        nbytes = 0
        for ptype in property_names:
            for name in property_names[ptype]:
                nbytes += data[ptype][name].nbytes
        mb_read = nbytes/(1024*1024)

        # Calculate read rate
        end = time.time()
        elapsed = end - start
        rate = mb_read / elapsed
        print("Read %.2f MB in %.2f seconds = %.2f MB/s" % (mb_read, elapsed, rate))

        return data

    def empty_mask(self):

        return np.zeros(self.dimension, dtype=np.bool)

    def mask_region(self, mask, pos_min, pos_max):

        pos_min = np.asarray(pos_min, dtype=float)
        pos_max = np.asarray(pos_max, dtype=float)
        imin = np.floor(pos_min/self.cell_size).astype(int)
        imax = np.floor(pos_max/self.cell_size).astype(int)
        for i in range(imin[0], imax[0]+1):
            ii = i % self.dimension[0]
            for j in range(imin[1], imax[1]+1):
                jj = j % self.dimension[1]
                for k in range(imin[2], imax[2]+1):
                    kk = k % self.dimension[2]
                    mask[i,j,k] = True
