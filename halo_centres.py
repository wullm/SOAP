#!/bin/env python

import h5py
import numpy as np

class SOCatalogue:

    def __init__(self, vr_basename):

        datasets = ("Xcminpot", "Ycminpot", "Zcminpot",
                    "Mass_tot", "R_size")
        data = {name : [] for name in datasets}

        # Read in position, mass and radius of each halo
        nr_files = 1
        file_nr = 0
        while file_nr < nr_files:
            fname = vr_basename + (".%d" % file_nr)
            with h5py.File(fname, "r") as infile:
                if file_nr == 0:
                    nr_files = infile["Num_of_files"][0]
                for name in data:
                    data[name].append(infile[name][...])
            file_nr +=1

        # Combine and store arrays
        for name in data:
            data[name] = np.concatenate(data[name])

        self.nr_halos = data["Mass_tot"].shape[0]
        self.mass_tot = data["Mass_tot"]
        self.r_size   = data["R_size"]
        self.centre = np.ndarray((self.nr_halos,3), dtype=data["Xcminpot"].dtype)
        self.centre[:,0] = data["Xcminpot"]
        self.centre[:,1] = data["Ycminpot"]
        self.centre[:,2] = data["Zcminpot"]
        self.index = np.arange(self.nr_halos, dtype=int)
        del data

    
