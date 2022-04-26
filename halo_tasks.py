#!/bin/env python

import unyt
import numpy as np
from dataset_names import mass_dataset
import shared_array
import time
import unyt

def process_single_halo(mesh, unit_registry, data, halo_prop_list, a, z,
                        critical_density, mean_density, boxsize, index,
                        centre, search_radius, read_radius, target_density):
    """
    This computes properties for one halo and runs on a single
    MPI rank. Result is a dict of properties of the form
    
    halo_result[property_name] = (unyt_array, description)

    where the property_name will be used as the HDF5 dataset name
    in the output and the units of the unyt_array determine the unit
    attributes.

    Two radii are passed in:

    search_radius is an initial guess at the radius we need to fall
    below the specified overdensity threshold. read_radius is the
    radius within which we have all of the particles. If we find that
    the density within read_radius is above the threshold, then we
    didn't read in a large enough region.
    """
    
    # Loop until we fall below the required density
    current_radius = search_radius
    while True:

        # Find the mass within the search radius
        mass_total = 0.0
        idx = {}
        for ptype in data:
            mass = data[ptype][mass_dataset(ptype)]
            pos = data[ptype]["Coordinates"]
            idx[ptype] = mesh[ptype].query_radius_periodic(centre, current_radius, pos, boxsize)
            mass_total += np.sum(mass.full[idx[ptype]], dtype=float)

        # If we have no target density, there's no need to iterate
        if target_density is None:
            break

        # Check if we reached the density threshold
        density = mass_total / (4./3.*np.pi*current_radius**3)
        if density <= target_density:
            # Reached the density threshold, so we're done
            break
        elif current_radius >= read_radius:
            # Still above target density and we've exceeded the region guaranteed to be read in
            raise Exception ("Read radius too small: r=%.2f, density ratio=%.2f" % (current_radius.value, density/target_density))
        else:
            # Need to increase the search radius and try again
            current_radius = min(current_radius*1.2, read_radius)

    # Extract particles in this halo
    halo_data = {}
    for ptype in data:
        halo_data[ptype] = {}
        for name in data[ptype]:
            halo_data[ptype][name] = data[ptype][name].full[idx[ptype],...]

    # Wrap coordinates to copy closest to the halo centre
    for ptype in halo_data:
        pos = halo_data[ptype]["Coordinates"]
        # Shift halo to box centre, wrap all particles into box, shift halo back
        offset = centre - 0.5*boxsize
        pos[:,:] = ((pos - offset) % boxsize) + offset

    # Compute properties of this halo        
    halo_result = {}
    for halo_prop in halo_prop_list:
        halo_result.update(halo_prop.calculate(index, unit_registry, critical_density, mean_density, a, z, centre, halo_data))

    # Add the halo index to the result set
    halo_result["index"] = (unyt.unyt_array(index, dtype=index.dtype, registry=unit_registry), "Index of this halo in the input catalogue")

    # Store search radius and density within that radius
    halo_result["search_radius"]            = (current_radius, "Search radius for property calculation")
    halo_result["density_in_search_radius"] = (density,        "Density within the search radius")
    halo_result["target_density"]           = (target_density, "Target density for property calculation")

    return halo_result


def process_halos(comm, unit_registry, data, mesh, halo_prop_list, a, z,
                  critical_density, mean_density, boxsize, indexes,
                  centres, search_radii, read_radii):

    # Compute density threshold at this redshift in comoving units:
    # This determines the size of the sphere we use for all other SO quantities.
    # We need to find the minimum density required for any of the halo property
    # calculations
    target_density = None
    for halo_prop in halo_prop_list:
        # Ensure target density is no greater than mean density multiple
        if halo_prop.mean_density_multiple is not None:
            density = halo_prop.mean_density_multiple*mean_density
            if target_density is None or density < target_density:
                target_density = density
        # Ensure target density is no greater than critical density multiple
        if halo_prop.critical_density_multiple is not None:
            density = halo_prop.critical_density_multiple*critical_density
            if target_density is None or density < target_density:
                target_density = density

    # Allocate shared storage for a single integer and initialize to zero
    if comm.Get_rank() == 0:
        local_shape = (1,)
    else:
        local_shape = (0,)
    next_task = shared_array.SharedArray(local_shape, np.int64, comm)
    if comm.Get_rank() == 0:
        next_task.full[0] = 0
    next_task.sync()

    # Start the clock
    comm.barrier()
    t0_all = time.time()

    # Loop until all halos are done
    result_arrays = {}
    nr_halos = len(indexes.full)
    nr_done_this_rank = 0
    nr_halos_this_rank_guess = int(nr_halos / comm.Get_size() * 1.5)
    task_time = 0.0
    while True:

        # Get a task by atomic incrementing the counter. Don't know how to do
        # an atomic fetch and add in python, so will use MPI RMA calls!
        task_to_do = np.ndarray(1, dtype=np.int64)
        one = np.ones(1, dtype=np.int64)
        next_task.win.Lock(0)
        next_task.win.Fetch_and_op(one, task_to_do, 0)
        next_task.win.Unlock(0)
        task_to_do = int(task_to_do)

        # Execute the task, if there's one left
        if task_to_do < nr_halos:
            t0_task = time.time()

            # Fetch the results for this particular halo
            results = process_single_halo(mesh, unit_registry, data, halo_prop_list, a, z,
                                          critical_density, mean_density, boxsize,
                                          indexes.full[task_to_do], centres.full[task_to_do,:],
                                          search_radii.full[task_to_do], read_radii.full[task_to_do],
                                          target_density)

            # Loop over properties which were calculated
            for result_name, (result_data, result_description) in results.items():

                # Create a new array to store this property if necessary
                if result_name not in result_arrays:
                    shape = (nr_halos_this_rank_guess,) + result_data.shape
                    # need to ensure we don't pass a unyt_quantity to empty_like
                    # because unyt_quantities must be scalars only
                    arr = np.empty_like(unyt.unyt_array(result_data, registry=unit_registry), shape=shape)
                    result_arrays[result_name] = [arr, result_description]

                # Find the array to store this result
                result_array, result_description = result_arrays[result_name]

                # Ensure the array is large enough
                if nr_done_this_rank >= result_array.shape[0]:
                    new_shape = list(result_array.shape)
                    new_shape[0] *= 2
                    new_result_array = np.empty_like(unyt.unyt_array(result_array, registry=unit_registry), shape=new_shape)
                    new_result_array[0:result_array.shape[0],...] = result_array[...]
                    result_array = new_result_array
                    result_arrays[result_name] = [result_array, result_description]
                    
                # Store this property for this halo to the output array
                result_array[nr_done_this_rank,...] = result_data

            # Count halos processed on this rank
            nr_done_this_rank += 1
                    
            t1_task = time.time()
            task_time += (t1_task-t0_task)
        else:
            break

    # Resize output arrays, since they may have been allocated larger than needed
    for name in result_arrays:
        result_arrays[name][0] = result_arrays[name][0][:nr_done_this_rank,...]

    # Free the shared task counter
    next_task.free()

    # Stop the clock
    comm.barrier()
    t1_all = time.time()
    
    return result_arrays, t1_all-t0_all, task_time
