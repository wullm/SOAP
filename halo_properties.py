#!/bin/env python

import numpy as np

from dataset_names import mass_dataset

class HaloProperty:
    def __init__(self):
        pass

class SOMasses(HaloProperty):
    
    # Arrays which must be read in for this calculation
    particle_properties = {
        "PartType0" : ["Coordinates", "Masses"],
        "PartType1" : ["Coordinates", "Masses"],
        "PartType4" : ["Coordinates", "Masses"],
        "PartType5" : ["Coordinates", "DynamicalMasses"]
    }

    def calculate(self, cosmo, a, z, centre, data):
        """
        Compute spherical masses and overdensities for a halo

        cosmo  - astropy cosmology object
        a      - expansion factor
        z      - redshift
        centre - coordinates of the halo centre
        data   - contains particle data. E.g. data["PartType1"]["Coordinates"]
                 has the particle coordinates for type 1

        Input particle data arrays are astropy Quantities.
        """

        result = {}

        # Make an array of particle masses and radii
        mass = []
        radius = []
        for ptype in data:
            mass.append(data[ptype][mass_dataset(ptype)])
            pos = data[ptype]["Coordinates"] - centre[None,:]
            r = np.sqrt(np.sum(pos**2, axis=1))
            radius.append(r)
        mass = np.concatenate(mass)
        radius = np.concatenate(radius)

        # Sort by radius
        order = np.argsort(radius)
        mass = mass[order]
        radius = radius[order]

        # Compute density within radius of each particle
        cumulative_mass = np.cumsum(mass)
        density = cumulative_mass / (4./3.*np.pi*radius**3)

        # Find critical density in comoving coordinates
        critical_density = cosmo.critical_density(z)*(a**3.0)

        # Find smallest radius where the density is below the threshold
        i = np.argmax(density < 200*critical_density)

        # Return value should be a dict containing astropy Quantities (i.e. with units)
        # and descriptions. The dict keys will be used as HDF5 dataset names in the output.
        return {
            "r_200_crit" : (radius[i],          "Radius within which the density is 200 times the mean"),
            "m_200_crit" : (cumulative_mass[i], "Mass within a sphere with density 200 times the mean"),
        }
