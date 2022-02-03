#!/bin/env python

import numpy as np
import scipy.spatial
import astropy.units

import matplotlib.pyplot as plt

from dataset_names import mass_dataset

def box_wrap(pos, ref_pos, boxsize):
    shift = ref_pos[None,:] - 0.5*boxsize
    return (pos - shift) % boxsize + shift

def compute_so_properties(cellgrid, centres, radii, pos_min, pos_max,
                          halo_prop_list):
    """
    This function finds all of the particles in a sphere around each halo
    and carries out the calculations specified in halo_prop_list.
    """
    cosmo = cellgrid.cosmology
    a = cellgrid.a
    z = cellgrid.z
    boxsize = cellgrid.boxsize
    ref_pos = (pos_min+pos_max)/2

    # Find all particle properties we need to read in:
    # For each particle type this is the union of the quantities
    # needed for each calculation.
    properties = {}
    for halo_prop in halo_prop_list:
        for ptype in halo_prop.particle_properties:
            if ptype not in properties:
                properties[ptype] = set()
            properties[ptype] = properties[ptype].union(halo_prop.particle_properties[ptype])

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

    # Dict to store the results:
    # Will eventually have one array for each quantity to calculate.
    result_arrays = {}

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
                # Call functions to compute halo properties for this halo
                halo_result = {}
                for halo_prop in halo_prop_list:
                    halo_result.update(halo_prop.calculate(cosmo, a, z, centres[halo_nr,:], data))
                # Store results
                for name, (value, description) in halo_result.items():
                    # If this is the first time we computed this quantity, allocate a new output array
                    if name not in result_arrays:
                        arr = astropy.units.Quantity(-np.ones(nr_halos, dtype=float), unit=value.unit)
                        result_arrays[name] = (arr, description)
                    # Store the result for this halo into the output array
                    result_arrays[name][0][halo_nr] = value
            else:
                # Need to increase the search radius for this one and try again
                rsearch[halo_nr] *= 1.5
                del idx

        # Remove completed halos from todo list
        to_do = to_do[to_do>=0]

    # Return the halo properties from this task
    return result_arrays
