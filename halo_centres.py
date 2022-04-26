#!/bin/env python

import os.path
import h5py
import numpy as np
import unyt
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.gather_array as g

class SOCatalogue:

    def __init__(self, comm, vr_basename, a_unit, registry, boxsize):

        comm_rank = comm.Get_rank()

        # Get SWIFT's definition of physical and comoving Mpc units
        pMpc = unyt.Unit("pMpc", registry=registry)
        cMpc = unyt.Unit(a_unit*pMpc, registry=registry)

        # Get expansion factor as a float
        a = a_unit.base_value

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

        # Compute conversion factors to comoving Mpc (no h)
        comoving_or_physical = int(units["Comoving_or_Physical"])
        length_unit_to_kpc = float(units["Length_unit_to_kpc"])
        h = float(siminfo["h_val"])
        if comoving_or_physical == 0:
            # Physical units with no h factor
            length_conversion = (1.0/a) * length_unit_to_kpc / 1000.0 # to comoving Mpc
        else:
            # Comoving 1/h units
            length_conversion = h * length_unit_to_kpc / 1000.0 # to comoving Mpc

        # Convert units
        local_cofm *= length_conversion
        local_cofp *= length_conversion
        local_r_size *= length_conversion

        # Add units to local arrays now that everything is in comoving Mpc
        local_cofm = unyt.unyt_array(local_cofm, units=cMpc)
        local_cofp = unyt.unyt_array(local_cofp, units=cMpc)
        local_r_size = unyt.unyt_array(local_r_size, units=cMpc)

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
        dist = np.sqrt(np.sum(dist**2, axis=1))

        # Store the initial search radius
        local_search_radius = (local_r_size*1.01 + dist)

        # Compute radius to read in about each halo:
        # this is the maximum radius we'll search to reach the required overdensity
        local_read_radius = local_search_radius.copy()
        min_radius = 5.0*cMpc
        ind = local_read_radius < min_radius
        local_read_radius[ind] = min_radius

        # Store the length unit
        length_unit = local_cofm.units

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
            self.search_radius = unyt.unyt_array(search_radius, units=length_unit)
            self.read_radius = unyt.unyt_array(read_radius, units=length_unit)
            self.centre = unyt.unyt_array(centre, units=length_unit)
            self.index = np.arange(self.nr_halos, dtype=int)

            # # Reduce number of halos for testing
            # self.nr_halos = 100
            # self.search_radius = self.search_radius[:self.nr_halos]
            # self.read_radius   = self.read_radius[:self.nr_halos]
            # self.centre        = self.centre[:self.nr_halos,:]
            # self.index         = self.index[:self.nr_halos]

