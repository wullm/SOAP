#!/bin/env python

import numpy as np
from dataset_names import mass_dataset
import astropy.units as u
import shared_array
import astropy.units
import time


def process_single_halo(mesh, data, halo_prop_list, a, z, cosmo,
                        index, centre, initial_search_radius):
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
    search_radius = initial_search_radius
    while True:

        # Find the mass within the search radius
        mass_total = 0.0
        idx = {}
        for ptype in data:
            mass = data[ptype][mass_dataset(ptype)]
            pos = data[ptype]["Coordinates"]
            idx[ptype] = mesh[ptype].query_radius(centre, search_radius, pos)
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
        halo_result.update(halo_prop.calculate(cosmo, a, z, centre, halo_data))

    # Add the halo index to the result set
    halo_result["index"] = (u.Quantity(index, unit=None), "Index of this halo in the input catalogue")

    return halo_result


def process_halos(comm, data, mesh, halo_prop_list, a, z, cosmo,
                  indexes, centres, radii):
    
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
            results.append(process_single_halo(mesh, data, halo_prop_list, a, z, cosmo,
                                               indexes.full[task_to_do], centres.full[task_to_do,:],
                                               radii.full[task_to_do]))
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
                arr = astropy.units.Quantity(-np.ones(nr_halos, dtype=float), unit=value.unit)
                result_arrays[name] = (arr, description)
            result_arrays[name][0][halo_nr] = value

    # Stop the clock
    comm.barrier()
    t1_all = time.time()

    # Measure dead time (i.e. time not doing halo calculations)
    total_time = comm.allreduce(t1_all - t0_all)
    dead_time  = total_time - comm.allreduce(task_time)
    
    return result_arrays, dead_time/total_time
