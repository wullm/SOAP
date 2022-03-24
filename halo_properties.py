#!/bin/env python

import numpy as np

from dataset_names import mass_dataset
import astropy.units as u

class HaloProperty:
    def __init__(self):
        pass

class SOMasses(HaloProperty):
    
    # Arrays which must be read in for this calculation.
    # Note that if there are no particles of a given type in the
    # snapshot, that type will not be read in and will not have
    # an entry in the data argument to calculate(), below.
    # (E.g. gas, star or BH particles in DMO runs)
    particle_properties = {
        "PartType0" : ["Coordinates", "Masses"],
        "PartType1" : ["Coordinates", "Masses"],
        "PartType4" : ["Coordinates", "Masses"],
        "PartType5" : ["Coordinates", "DynamicalMasses"]
    }

    # This specifies how large a sphere is read in:
    # Will ensure we have a sphere with a mean density less than
    # or equal to the minimum of these densities.
    mean_density_multiple     = 200.0
    critical_density_multiple = 200.0

    def calculate(self, index, cosmo, a, z, centre, data):
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
        nr_parts = mass.shape[0]

        # Sort by radius
        order = np.argsort(radius)
        mass = mass[order]
        radius = radius[order]

        # Compute density within radius of each particle
        cumulative_mass = np.cumsum(mass, dtype=np.float64)
        density = cumulative_mass / (4./3.*np.pi*radius**3)

        # Find critical density in comoving coordinates
        critical_density = cosmo.critical_density(z)*(a**3.0)

        # Check if we ever reach the density threshold
        if nr_parts > 1 and np.any(density > 200*critical_density):
            # Find smallest radius where the density is below the threshold,
            # ignoring the first particle
            i = np.argmax(density[1:] < 200*critical_density)
            m200crit = cumulative_mass[1:][i]
            r200crit = radius[1:][i]
        else:
            # Below threshold at all radii. Need to return zero with correct units attached.
            m200crit = u.Quantity(0, dtype=cumulative_mass.dtype, unit=cumulative_mass.unit)
            r200crit = u.Quantity(0, dtype=radius.dtype, unit=radius.unit)

        # Return value should be a dict containing astropy Quantities (i.e. with units)
        # and descriptions. The dict keys will be used as HDF5 dataset names in the output.
        return {
            "r_200_crit" : (r200crit, "Radius within which the density is 200 times the mean"),
            "m_200_crit" : (m200crit, "Mass within a sphere with density 200 times the mean"),
        }


class CentreOfMass(HaloProperty):
    
    # Arrays which must be read in for this calculation.
    # Note that if there are no particles of a given type in the
    # snapshot, that type will not be read in and will not have
    # an entry in the data argument to calculate(), below.
    # (E.g. gas, star or BH particles in DMO runs)
    particle_properties = {
        "PartType0" : ["Coordinates", "Masses", "GroupNr_all"],
        "PartType1" : ["Coordinates", "Masses", "GroupNr_all"],
        "PartType4" : ["Coordinates", "Masses", "GroupNr_all"],
        "PartType5" : ["Coordinates", "DynamicalMasses", "GroupNr_all"]
    }

    # This specifies how large a sphere is read in:
    # Will ensure we have a sphere with a mean density less than
    # or equal to the minimum of these densities.
    mean_density_multiple     = None
    critical_density_multiple = None

    def calculate(self, index, cosmo, a, z, centre, data):
        """
        Compute centre of mass of bound particles

        cosmo  - astropy cosmology object
        a      - expansion factor
        z      - redshift
        centre - coordinates of the halo centre
        data   - contains particle data. E.g. data["PartType1"]["Coordinates"]
                 has the particle coordinates for type 1

        Input particle data arrays are astropy Quantities.
        """
        
        cofm = None
        mtot = None
        nr_part = 0

        # Loop over particle types
        for ptype in data:

            # Find position and mass of particles in the group
            grnr = data[ptype]["GroupNr_all"]
            in_halo = (grnr==index)
            pos  = data[ptype]["Coordinates"][in_halo,:]
            mass = data[ptype][mass_dataset(ptype)][in_halo]

            # Accumulate total mass of particles
            m = np.sum(mass, dtype=float)
            if mtot is None:
                mtot = m
            else:
                mtot += m

            # Accumulate position*mass for particles in this group
            pos_mass = np.sum(pos*mass[:,None], axis=0, dtype=float)
            if cofm is None:
                cofm = pos_mass
            else:
                cofm += pos_mass
        
            # Accumulate total number of particles
            nr_part += pos.shape[0]

        # Compute centre of mass
        cofm /= mtot

        return {
            "CentreOfMass" : (cofm, "Centre of mass of particles in the group"),
            "Mass"         : (mtot, "Total mass of particles in this group"),
            "NrParticles"  : (u.Quantity(nr_part, unit=None, dtype=int), "Number of bound or unbound particles in this group"),
        }
