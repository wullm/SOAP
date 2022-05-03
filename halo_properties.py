#!/bin/env python

import numpy as np
import unyt

from dataset_names import mass_dataset

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

    def calculate(self, unit_registry, critical_density, mean_density, a, z,
                  input_halo, data, halo_result):
        """
        Compute spherical masses and overdensities for a halo

        unit_registry    - unyt unit registry defining simulation units
        critical_density - critical density from the snapshot, as unyt_quantity
        mean_density     - mean density from the snapshot, as unyt_quantity
        a                - expansion factor
        z                - redshift
        input_halo       - dict with halo properties passed in from VR (see
                           halo_centres.py)
        data             - contains particle data. E.g. data["PartType1"]["Coordinates"]
                           has the particle coordinates for type 1
        halo_result      - dict with halo properties computed so far. Properties
                           computed here should be added to halo_result.

        Input particle data arrays are unyt_arrays.
        """

        # Look up VR centre of potential for this halo
        centre = input_halo["cofp"]

        # Make an array of particle masses and radii
        mass = []
        radius = []
        for ptype in data:
            mass.append(data[ptype][mass_dataset(ptype)])
            pos = data[ptype]["Coordinates"] - centre[None,:]
            r = np.sqrt(np.sum(pos**2, axis=1))
            radius.append(r)
        mass = unyt.array.uconcatenate(mass)
        radius = unyt.array.uconcatenate(radius)
        nr_parts = mass.shape[0]

        # Sort by radius
        order = np.argsort(radius)
        mass = mass[order]
        radius = radius[order]
        cumulative_mass = np.cumsum(mass, dtype=np.float64)

        # Compute density within radius of each particle.
        # Will need to skip any at zero radius.
        nskip = 0
        while nskip < len(radius) and radius[nskip] == 0:
            nskip += 1
        radius  = radius[nskip:]
        cumulative_mass = cumulative_mass[nskip:]
        nr_parts = len(radius)
        density = cumulative_mass / (4./3.*np.pi*radius**3)

        # Check if we ever reach the density threshold
        if nr_parts > 0 and np.any(density > 200*critical_density):
            # Find smallest radius where the density is below the threshold
            i = np.argmax(density < 200*critical_density)
            m200crit = cumulative_mass[i]
            r200crit = radius[i]
        else:
            # Below threshold at all radii. Need to return zero with correct units attached.
            m200crit = unyt.unyt_array(0, dtype=cumulative_mass.dtype, units=cumulative_mass.units)
            r200crit = unyt.unyt_array(0, dtype=radius.dtype, units=radius.units)

        # Update this halo's properties
        halo_result.update({
            "r_200_crit" : (r200crit, "Radius within which the density is 200 times the mean"),
            "m_200_crit" : (m200crit, "Mass within a sphere with density 200 times the mean"),
        })


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

    def calculate(self, unit_registry, critical_density, mean_density, a, z,
                  input_halo, data, halo_result):
        """
        Compute centre of mass of bound particles

        unit_registry    - unyt unit registry defining simulation units
        critical_density - critical density from the snapshot, as unyt_quantity
        mean_density     - mean density from the snapshot, as unyt_quantity
        a                - expansion factor
        z                - redshift
        input_halo       - dict with halo properties passed in from VR (see
                           halo_centres.py)
        data             - contains particle data. E.g. data["PartType1"]["Coordinates"]
                           has the particle coordinates for type 1
        halo_result      - dict with halo properties computed so far. Properties
                           computed here should be added to halo_result.

        Input particle data arrays are unyt_arrays.
        """
        
        cofm = None
        mtot = None
        nr_part = 0

        # Look up array index of this halo in VR catalogue
        index = input_halo["index"]

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

        # Return number of particles
        nr_part = unyt.unyt_array(nr_part, dtype=int, registry=unit_registry)

        # Update the halo properties
        halo_result.update({
            "CentreOfMass" : (cofm,    "Centre of mass of particles in the group"),
            "Mass"         : (mtot,    "Total mass of particles in this group"),
            "NrParticles"  : (nr_part, "Number of bound or unbound particles in this group"),            
        })
