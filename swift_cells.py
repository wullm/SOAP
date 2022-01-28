#!/bin/env python

import numpy as np
import h5py

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
        with h5py.File(filename % 0, "r") as infile:
            
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
        cells_to_read = np.lexsort((cells_to_read["offset"], cells_to_read["file"]))

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

    def read(self, property_names, mask):
        """
        Read the requested properties from the cells where mask=True.

        property_names[ptype] contains the list of quantity names to
        read for particle type ptype.

        mask is a 3D boolean array which flags the cells to read.

        Returns a dict with the data arrays:

        data[ptype][property_name] contains property property_name
        for particle type ptype.
        """

        # Dict to store the result. Initially an empty list for
        # each quantity to read.
        data = {}
        for ptype in property_names:
            data[ptype] = {name : [] for name in property_names[ptype]}
        
        # Find ranges of particles to read from each file
        reads = {ptype : self.prepare_read(ptype, mask) for ptype in property_names}

        # Find union of file numbers to read for all particle types
        all_file_nrs = []
        for ptype in property_names:
            all_file_nrs += list(reads[ptype])
        all_file_nrs = np.unique(all_files)

        # Loop over files to read
        for file_nr in all_file_nrs:

            # Open this file
            filename = self.filename % file_nr
            infile = h5py.File(filename, "r")

            # Loop over particle types to read
            for ptype in property_names:

                # Check if we need to read from this file for this particle type
                if file_nr in reads[ptype]:

                    # Loop over quantities to read for this particle type
                    for name in property_names[ptype]:

                        # Read the chunks for this property
                        dataset = infile[ptype][name]
                        for (offset, count) in reads[ptype][file_nr]:
                            data[ptype][name].append(dataset[offset:offset+count,...])

            # Close the file
            infile.close()

        # Merge chunks from all files
        for ptype in property_names:
            for name in property_names[ptype]:
                data[ptype][name] = np.concatenate(data[ptype][name])

        return data
