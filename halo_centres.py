#!/bin/env python

import h5py
import numpy as np
import astropy.units

class SOCatalogue:

    def __init__(self, vr_basename, a, parsec_cgs, solar_mass_cgs):

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
                    units = dict(infile["UnitInfo"].attrs)
                    siminfo = dict(infile["SimulationInfo"].attrs)
                for name in data:
                    data[name].append(infile[name][...])
            file_nr +=1

        # Combine and store arrays
        for name in data:
            data[name] = np.concatenate(data[name])

        # Compute conversion factors to comoving Mpc and Msolar (no h in either)
        comoving_or_physical = int(units["Comoving_or_Physical"])
        length_unit_to_kpc = float(units["Length_unit_to_kpc"])
        mass_unit_to_solarmass = float(units["Mass_unit_to_solarmass"])
        h = float(siminfo["h_val"])
        if comoving_or_physical == 0:
            # Physical units with no h factor
            length_conversion = (1.0/a) * length_unit_to_kpc / 1000.0 # to comoving Mpc
            mass_conversion = mass_unit_to_solarmass # To solar masses
        else:
            # Comoving 1/h units
            length_conversion = h * length_unit_to_kpc / 1000.0 # to comoving Mpc
            mass_conversion = h * mass_unit_to_solarmass # To solar masses

        # Use SWIFT's defintions of parsec, solar mass
        mass_unit = astropy.units.g * solar_mass_cgs
        length_unit = astropy.units.cm * parsec_cgs

        # Read the data
        self.nr_halos = data["Mass_tot"].shape[0]
        self.mass_tot = astropy.units.Quantity(data["Mass_tot"] * mass_conversion, unit=mass_unit)
        self.r_size   = astropy.units.Quantity(data["R_size"] * length_conversion, unit=length_unit)
        self.centre   = astropy.units.Quantity(np.ndarray((self.nr_halos,3), dtype=data["Xcminpot"].dtype), unit=length_unit)
        self.centre[:,0] = data["Xcminpot"] * length_conversion * length_unit
        self.centre[:,1] = data["Ycminpot"] * length_conversion * length_unit
        self.centre[:,2] = data["Zcminpot"] * length_conversion * length_unit
        self.index = np.arange(self.nr_halos, dtype=int)
        del data

    
