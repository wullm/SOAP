#!/bin/env python

import sys
import h5py
import numpy as np


def update_vds_paths(filename, prefix):
    """
    Add the specified prefix to virtual datasets in a file
    """

    f = h5py.File(filename, "r+")

    # Get the names of all virtual datasets in the file
    paths = []
    def visit_path(name):
        obj = f[name]
        if isinstance(obj, h5py.Dataset) and obj.is_virtual:
            paths.append(name)
    f.visit(visit_path)

    # Loop over virtual datasets
    for path in paths:

        print(path)

        # Find the original virtual dataset
        dset = f[path]

        # Choose a path for the new virtual dataset
        tmp_path = path+".__tmp__"

        # Build the creation property list for the new dataset
        plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        for vs in dset.virtual_sources():
            lower, upper = vs.vspace.get_select_bounds()
            size = np.asarray(upper, dtype=int)-np.asarray(lower, dtype=int)+1
            src_space = h5py.h5s.create_simple(tuple(size))
            plist.set_virtual(vs.vspace, (prefix+vs.file_name).encode(), vs.dset_name.encode(), src_space)
        
        # Create the new dataset
        tmp_dset = h5py.h5d.create(f["/"].id, tmp_path.encode(), dset.id.get_type(), dset.id.get_space(), dcpl=plist)

        # Copy any attributes
        src_dset = f[path]
        dest_dset = f[tmp_path]
        for attr_name in src_dset.attrs:
            dest_dset.attrs[attr_name] = src_dset.attrs[attr_name]

        # Rename the new dataset
        del f[path]
        f[path] = f[tmp_path]
        del f[tmp_path]


if __name__ == "__main__":

    filename = sys.argv[1]
    prefix   = sys.argv[2]
    update_vds_paths(filename, prefix)
