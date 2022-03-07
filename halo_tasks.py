#!/bin/env python

import numpy as np
from dataset_names import mass_dataset

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
                idx[ptype] = mesh.query_radius(self.centre, search_radius, pos)
                mass_total += np.sum(mass[idx[ptype]], dtype=float)

            # Check if we reached the density threshold
            density = mass_total / (4./3.*np.pi*search_radius**3)
            if density < target_density:
                break
            else:
                search_radius *= 1.5

        # Find particles in this halo
        halo_data = {}
        for ptype in data:
            halo_data[ptype] = {}
            for name in data[ptype]:
                halo_data[ptype][name] = data[ptype][name][idx[ptype],...]

        # Compute properties of this halo        
        halo_result = {}
        for halo_prop in halo_prop_list:
            halo_result.update(halo_prop.calculate(cosmo, a, z, centre, halo_data))

        # Add the halo index to the result set
        halo_result["index"] = (self.index, "Index of this halo in the input catalogue")

        return halo_result
