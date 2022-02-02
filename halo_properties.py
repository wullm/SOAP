#!/bin/env python

import numpy as np
import scipy.spatial
import astropy.units

import matplotlib.pyplot as plt

def mass_dataset(ptype):
    if ptype == "PartType5":
        return "DynamicalMasses"
    else:
        return "Masses"

def box_wrap(pos, ref_pos, boxsize):
    shift = ref_pos[None,:] - 0.5*boxsize
    return (pos - shift) % boxsize + shift

def compute_so_properties(cellgrid, centres, radii, pos_min, pos_max):

    cosmo = cellgrid.cosmology
    a = cellgrid.a
    z = cellgrid.z
    boxsize = cellgrid.boxsize
    ref_pos = (pos_min+pos_max)/2

    properties = {}
    
    # Gas properties to read
    if "PartType0" in cellgrid.ptypes:
        properties["PartType0"] = ["Coordinates", "Masses"]

    # DM properties to read
    if "PartType1" in cellgrid.ptypes:
        properties["PartType1"] = ["Coordinates", "Masses"]

    # Star properties to read
    if "PartType4" in cellgrid.ptypes:
        properties["PartType4"] = ["Coordinates", "Masses"]

    # BH properties to read
    if "PartType5" in cellgrid.ptypes:
        properties["PartType5"] = ["Coordinates", "DynamicalMasses"]

    # Read in particles in the required region
    mask = cellgrid.empty_mask()
    cellgrid.mask_region(mask, pos_min, pos_max)
    data = cellgrid.read_masked_cells(properties, mask, verbose=False)

    # Count particles and return if there are none
    nr_parts = 0
    for ptype in data:
        name = mass_dataset(ptype)
        if name in data[ptype]:
            nr_parts += data[ptype][name].shape[0]
    if nr_parts == 0:
        return None

    # Do periodic shift of particles to copies nearest the reference point
    for ptype in data:
        if "Coordinates" in data[ptype]:
            data[ptype]["Coordinates"] = box_wrap(data[ptype]["Coordinates"], ref_pos, boxsize)

    # Build kd tree for each particle type
    tree = {}
    for ptype in properties:
        tree[ptype] = scipy.spatial.cKDTree(data[ptype]["Coordinates"], balanced_tree=False)

    # Compute density threshold at this redshift in comoving units:
    # This determines the size of the sphere we use for all other SO quantities.
    target_density = 50*cosmo.critical_density(z)*(a**3.0)

    # For each halo, find radius within which density falls below threshold.
    # Use R_size as initial guess.
    nr_halos = centres.shape[0]
    rsearch = radii.copy()

    # Dict to store the results
    result = {}

    # Iterate until we find a radius such that the mean density is < target_density for all halos
    to_do = np.arange(nr_halos, dtype=int)
    while len(to_do) > 0:

        # Loop over remaining halos
        for i, halo_nr in enumerate(to_do):

            # Find the mass within the search radius
            mass_total = 0.0
            idx = {}
            for ptype in data:
                mass = data[ptype][mass_dataset(ptype)]
                idx[ptype] = np.asarray((tree[ptype].query_ball_point(centres[halo_nr,:], rsearch[halo_nr])), dtype=int)
                mass_total += np.sum(mass[idx[ptype]], dtype=float)

            # Compute the mean density in the search radius
            density = mass_total / (4./3.*np.pi*rsearch[halo_nr]**3)
            if density < target_density:
                # Remove this halo from the to do list
                to_do[i] = -1
                # Find particles in this halo
                halo_data = {}
                for ptype in data:
                    halo_data[ptype] = {}
                    for name in data[ptype]:
                        halo_data[ptype][name] = data[ptype][name][idx[ptype],...]
                # Compute halo SO properties for this halo
                halo_result = halo_so_properties(cosmo, a, z, centres[halo_nr,:], halo_data)
                # Store results
                for name in halo_result:
                    if name not in result:
                        result[name] = astropy.units.Quantity(-np.ones(nr_halos, dtype=float), unit=halo_result[name].unit)
                    result[name][halo_nr] = halo_result[name]                
            else:
                # Need to increase the search radius for this one and try again
                rsearch[halo_nr] *= 1.5
                del idx

        # Remove completed halos from todo list
        to_do = to_do[to_do>=0]

    # Return the halo properties from this task
    return result


def halo_so_properties(cosmo, a, z, centre, data):
    """
    Compute SO properties for a halo

    data[ptype][name] contains property 'name' for particle
    type ptype for all particles in a sphere around this halo.
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
    result["r_200_crit"] = radius[i]
    result["m_200_crit"] = cumulative_mass[i]

    return result
