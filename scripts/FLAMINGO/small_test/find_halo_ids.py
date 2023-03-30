#!/bin/env python3
#
# Find IDs of halos in a corner of the simulation box
#
# Run with e.g. `python3 ./find_halo_ids.py L1000N1800/HYDRO_FIDUCIAL 77 10`
#
import sys
import numpy as np
import h5py


def find_halo_ids(sim, snap_nr, boxsize):
    
    soap_file=f"/cosma8/data/dp004/flamingo/Runs/{sim}/SOAP/halo_properties_{snap_nr:04d}.hdf5"
    with h5py.File(soap_file, "r") as f:
        pos = f['VR']['CentreOfPotential'][...]
        ids = f['VR']['ID'][...]
    ind = np.all(pos < boxsize, axis=1)
    return ids[ind]


if __name__ == "__main__":

    sim     = sys.argv[1]
    snap_nr = int(sys.argv[2])
    boxsize = float(sys.argv[3])

    ids = find_halo_ids(sim, snap_nr, boxsize)

    id_list = " ".join([str(i) for i in ids])
    print(id_list)
