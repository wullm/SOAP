#!/bin/env python

import os.path
import h5py
import numpy as np
import astropy.units
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.gather_array as g


class SOCatalogue:

    def __init__(self, comm, vr_basename, a, parsec_cgs, solar_mass_cgs, boxsize):

        comm_rank = comm.Get_rank()

        # Here we need to read the centre of mass AND potential minimum:
        # The radius R_size about (Xc, Yc, Zc) contains all particles which
        # belong to the group. But we want to compute spherical overdensity
        # quantities about the potential minimum.
        datasets = ("Xcminpot", "Ycminpot", "Zcminpot",
                    "Xc", "Yc", "Zc", "R_size")

        # Check for single file VR output - will prefer filename without
        # extension if both are present
        vr_basename += ".properties"
        if comm_rank == 0:
            if os.path.exists(vr_basename):
                filenames = vr_basename
            else:
                filenames = vr_basename+".%(file_nr)d"
        else:
            filenames = None
        filenames = comm.bcast(filenames)

        # Read in positions and radius of each halo, distributed over all MPI ranks
        mf = phdf5.MultiFile(filenames, file_nr_dataset="Num_of_files")
        data = mf.read(datasets)

        # Combine positions into one array each
        local_cofm = np.column_stack((data["Xc"], data["Yc"], data["Zc"]))
        local_cofp = np.column_stack((data["Xcminpot"], data["Ycminpot"], data["Zcminpot"]))
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
        local_cofm *= length_conversion
        local_cofp *= length_conversion
        local_r_size *= length_conversion

        # Add units to local arrays
        local_cofm = astropy.units.Quantity(local_cofm, unit=length_unit, copy=False)
        local_cofp = astropy.units.Quantity(local_cofp, unit=length_unit, copy=False)
        local_r_size = astropy.units.Quantity(local_r_size, unit=length_unit, copy=False)

        #
        # Compute initial search radius for each halo:
        #
        # Need to ensure that our radius about the potential minimum
        # includes all particles within r_size of the centre of mass.
        #
        # Find distance from centre of mass to centre of potential,
        # taking the periodic box into account
        dist = np.abs(local_cofp - local_cofm)
        for dim in range(3):
            need_wrap = dist[:,dim] > 0.5*boxsize
            dist[need_wrap, dim] = boxsize - dist[need_wrap, dim]
        dist = np.linalg.norm(dist)

        # Store the initial search radius
        local_search_radius = (local_r_size*1.01 + dist)

        # Compute radius to read in about each halo:
        # this is the maximum radius we'll search to reach the required overdensity
        local_read_radius = local_search_radius.copy()
        min_radius = 5.0*length_unit
        ind = local_read_radius < min_radius
        local_read_radius[ind] = min_radius

        # Free some arrays we don't need
        del dist
        del local_cofm
        del local_r_size

        # Gather arrays on rank zero.
        # Will strip units to communicate the arrays then add them back afterwards.
        search_radius = g.gather_array(local_search_radius.value)
        read_radius = g.gather_array(local_read_radius.value)
        del local_search_radius
        del local_read_radius
        centre = g.gather_array(local_cofp.value)
        del local_cofp
        if comm_rank == 0:
            self.nr_halos = len(search_radius)
            self.search_radius = astropy.units.Quantity(search_radius, unit=length_unit, copy=False)
            self.read_radius = astropy.units.Quantity(read_radius, unit=length_unit, copy=False)
            self.centre = astropy.units.Quantity(centre, unit=length_unit, copy=False)
            self.index = np.arange(self.nr_halos, dtype=int)

