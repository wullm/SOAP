#!/bin/env python

import numpy as np
from dataset_names import mass_dataset
import astropy.units as u

class HaloTask:

    def __init__(self, index, centre, initial_search_radius):
        self.index = index
        self.centre = centre
        self.initial_search_radius = initial_search_radius

    def __call__(self, mesh, data, halo_prop_list, a, z, cosmo):
        """
        This computes properties for one halo and runs on a single
        MPI rank. Result is a dict of properties of the form

        halo_result[property_name] = (astropy quantity, description)

        where the property_name will be used as the HDF5 dataset name
        in the output and the units of the astropy Quantity determine
        the unit attributes.
        """

        # Compute density threshold at this redshift in comoving units:
        # This determines the size of the sphere we use for all other SO quantities.
        target_density = 50*cosmo.critical_density(z)*(a**3.0)
        
        # Loop until search radius is large enough
        search_radius = self.initial_search_radius
        while True:

            # Find the mass within the search radius
            mass_total = 0.0
            idx = {}
            for ptype in data:
                mass = data[ptype][mass_dataset(ptype)]
                pos = data[ptype]["Coordinates"]
                idx[ptype] = mesh[ptype].query_radius(self.centre, search_radius, pos)
                mass_total += np.sum(mass.full[idx[ptype]], dtype=float)

            # Check if we reached the density threshold
            density = mass_total / (4./3.*np.pi*search_radius**3)
            if density < target_density:
                break
            else:
                search_radius *= 1.2
                del idx

        # Find particles in this halo
        halo_data = {}
        for ptype in data:
            halo_data[ptype] = {}
            for name in data[ptype]:
                halo_data[ptype][name] = data[ptype][name].full[idx[ptype],...]

        # Compute properties of this halo        
        halo_result = {}
        for halo_prop in halo_prop_list:
            halo_result.update(halo_prop.calculate(cosmo, a, z, self.centre, halo_data))

        # Add the halo index to the result set
        halo_result["index"] = (u.Quantity(self.index, unit=None), "Index of this halo in the input catalogue")

        return halo_result
