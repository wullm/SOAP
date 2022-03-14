#!/bin/env python

import numpy as np
from dataset_names import mass_dataset
import astropy.units as u
import shared_array
import astropy.units
import time


def process_single_halo(mesh, data, halo_prop_list, a, z, cosmo,
                        boxsize, max_halo_radius, index, centre,
                        initial_search_radius):
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
    # We need to find the minimum density required for any of the halo property
    # calculations
    target_density = None
    critical_density = cosmo.critical_density(z)*(a**3.0)
    mean_density = critical_density * cosmo.Om(z)
    for halo_prop in halo_prop_list:
        # Ensure target density is no greater than mean density multiple
        density = halo_prop.mean_density_multiple*mean_density
        if target_density is None or density < target_density:
            target_density = density
        # Ensure target density is no greater than critical density multiple
        density = halo_prop.critical_density_multiple*critical_density
        if target_density is None or density < target_density:
            target_density = density
    assert target_density is not None

    # Loop until search radius is large enough
    search_radius = initial_search_radius
    while True:

        # Find the mass within the search radius
        mass_total = 0.0
        idx = {}
        for ptype in data:
            mass = data[ptype][mass_dataset(ptype)]
            pos = data[ptype]["Coordinates"]
            idx[ptype] = mesh[ptype].query_radius_periodic(centre, search_radius, pos, boxsize)
            mass_total += np.sum(mass.full[idx[ptype]], dtype=float)

        # Check if we reached the density threshold
        density = mass_total / (4./3.*np.pi*search_radius**3)
        if density < target_density:
            break
        else:
            search_radius *= 1.2
            del idx

    # Check we didn't exceed the region we read in
    if search_radius > max_halo_radius:
        raise Exception("Search radius for halo is too large!")

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
        halo_result.update(halo_prop.calculate(cosmo, a, z, centre, halo_data))

    # Add the halo index to the result set
    halo_result["index"] = (u.Quantity(index, unit=None, dtype=np.int64), "Index of this halo in the input catalogue")

    # Store search radius and density within that radius
    halo_result["search_radius"] = (search_radius, "Search radius for property calculation")
    halo_result["density_in_search_radius"] = (density, "Density within the search radius")
    halo_result["target_density"] = (target_density, "Target density for property calculation")

    return halo_result


def process_halos(comm, data, mesh, halo_prop_list, a, z, cosmo,
                  boxsize, max_halo_radius, indexes, centres, radii):
    
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
    results = []
    nr_halos = len(indexes.full)
    task_time = 0.0
    while True:

        # Get a task by atomic incrementing the counter
        task_to_do = np.ndarray(1, dtype=np.int64)
        one = np.ones(1, dtype=np.int64)
        next_task.win.Lock(0)
        next_task.win.Fetch_and_op(one, task_to_do, 0)
        next_task.win.Unlock(0)
        task_to_do = int(task_to_do)

        # Execute the task, if there's one left
        if task_to_do < nr_halos:
            t0_task = time.time()
            results.append(process_single_halo(mesh, data, halo_prop_list, a, z, cosmo, boxsize,
                                               max_halo_radius, indexes.full[task_to_do],
                                               centres.full[task_to_do,:], radii.full[task_to_do]))
            t1_task = time.time()
            task_time += (t1_task-t0_task)
        else:
            break

    # Free the shared task counter
    next_task.free()

    # Combine task results into arrays
    nr_halos = len(results)
    result_arrays = {}
    for halo_nr, result in enumerate(results):
        for name, (value, description) in result.items():
            if name not in result_arrays:
                arr = np.empty_like(value, shape=nr_halos)
                result_arrays[name] = (arr, description)
            result_arrays[name][0][halo_nr] = value

    # Stop the clock
    comm.barrier()
    t1_all = time.time()
    
    return result_arrays, t1_all-t0_all, task_time
