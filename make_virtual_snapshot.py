#!/bin/env python

import os.path
import numpy as np
import h5py
import argparse
import shutil


def make_virtual_snapshot(snapshot, membership, output_file):
    """
    Given a FLAMINGO snapshot and VR group membership files,
    create a new virtual snapshot with group info.
    """
    
    # Copy the input virtual snapshot to the output
    shutil.copyfile(snapshot, output_file)

    # Open the output file
    outfile = h5py.File(output_file, "r+")

    # Loop over input membership files to get dataset shapes
    file_nr = 0
    filenames = []
    shapes = []
    dtype=None
    while True:
        filename = membership % {"file_nr" : file_nr}
        if os.path.exists(filename):
            filenames.append(filename)
            with h5py.File(filename, "r") as infile:
                shape = {}
                for ptype in range(7):
                    name = f"PartType{ptype}"
                    if name in infile:
                        shape[ptype] = infile[name]["GroupNr_all"].shape
                        if dtype is None:
                            dtype = infile[name]["GroupNr_all"].dtype
                shapes.append(shape)
        else:
            break
        file_nr += 1
    if file_nr == 0:
        raise IOError(f"Failed to find files matching: {membership}")

    # Loop over particle types in the output
    for ptype in range(7):
        name = f"PartType{ptype}"
        if name in outfile:
            # Create virtual layout for new datasets
            nr_parts = sum([shape[ptype][0] for shape in shapes])
            full_shape = (nr_parts,)
            layout_grnr_all   = h5py.VirtualLayout(shape=full_shape, dtype=dtype)
            layout_grnr_bound = h5py.VirtualLayout(shape=full_shape, dtype=dtype)
            layout_rank_bound = h5py.VirtualLayout(shape=full_shape, dtype=dtype)
            # Loop over input files
            offset = 0
            for (filename, shape) in zip(filenames, shapes):
                count = shape[ptype][0]
                layout_grnr_all[offset:offset+count]   = h5py.VirtualSource(filename, f'PartType{ptype}/GroupNr_all', shape=shape[ptype])
                layout_grnr_bound[offset:offset+count] = h5py.VirtualSource(filename, f'PartType{ptype}/GroupNr_bound', shape=shape[ptype])
                layout_rank_bound[offset:offset+count] = h5py.VirtualSource(filename, f'PartType{ptype}/Rank_bound', shape=shape[ptype])
                offset += count
            # Create the virtual datasets
            outfile.create_virtual_dataset(f'PartType{ptype}/GroupNr_all', layout_grnr_all,   fillvalue=-999)
            outfile.create_virtual_dataset(f'PartType{ptype}/GroupNr_bound', layout_grnr_bound, fillvalue=-999)
            outfile.create_virtual_dataset(f'PartType{ptype}/Rank_bound', layout_rank_bound, fillvalue=-999)
    
    # Done
    outfile.close()
