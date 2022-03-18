#!/bin/env python

import os.path
import h5py
import numpy as np
import astropy.units
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.gather_array as g


class SOCatalogue:

    def __init__(self, comm, vr_basename, a, parsec_cgs, solar_mass_cgs):

        comm_rank = comm.Get_rank()

        datasets = ("Xcminpot", "Ycminpot", "Zcminpot", "R_size")

        # Check for single file VR output - will prefer filename without
        # extension if both are present
        if comm_rank == 0:
            if os.path.exists(vr_basename):
                filenames = vr_basename
            else:
                filenames = vr_basename+".%(file_nr)d"
        else:
            filenames = None
        filenames = comm.bcast(filenames)

        # Read in position and radius of each halo, distributed over all MPI ranks
        mf = phdf5.MultiFile(filenames, file_nr_dataset="Num_of_files")
        data = mf.read(datasets)
        
        # Combine positions into one array
        x = data["Xcminpot"]
        y = data["Ycminpot"]
        z = data["Zcminpot"]
        local_centre = np.ndarray((x.shape[0], 3), dtype=x.dtype)
        local_centre[:,0] = x
        local_centre[:,1] = y
        local_centre[:,2] = z
        local_r_size = data["R_size"]

        # Extract unit information from the first file
        if comm_rank == 0:
            filename = filenames % {"file_nr" : 0}
            with h5py.File(filename, "r") as infile:
                units = dict(infile["UnitInfo"].attrs)
                siminfo = dict(infile["SimulationInfo"].attrs)
        else:
            units = None
            siminfo = None
        units, siminfo = comm.bcast((units, siminfo))

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
        length_unit = astropy.units.cm * 1.0e6 * parsec_cgs

        # Convert units
        local_centre   *= length_conversion
        local_r_size   *= length_conversion

        # Gather arrays on rank zero
        self.r_size   = g.gather_array(local_r_size)
        self.centre   = g.gather_array(local_centre)

        # Add units
        if comm_rank == 0:
            self.nr_halos = self.r_size.shape[0]
            self.index    = np.arange(self.nr_halos, dtype=int)
            self.r_size   = astropy.units.Quantity(self.r_size, unit=length_unit, copy=False)
            self.centre   = astropy.units.Quantity(self.centre, unit=length_unit, copy=False)
