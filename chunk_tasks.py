#!/bin/env python

import numpy as np
import astropy.units
from mpi4py import MPI

import task_queue
import shared_mesh
import halo_tasks
from dataset_names import mass_dataset


def box_wrap(pos, ref_pos, boxsize):
    shift = ref_pos[None,:] - 0.5*boxsize
    return (pos - shift) % boxsize + shift


class ChunkTaskList:
    """
    Stores a list of ChunkTasks to be executed.
    """
    def __init__(self, cellgrid, so_cat, search_radius, cells_per_task,
                 halo_prop_list):
                
        # Find size of volume associated with each task
        task_size = cellgrid.cell_size[0]*cells_per_task
        
        # For each centre, determine integer coords in task grid
        ipos = np.floor(so_cat.centre / task_size).value.astype(int)

        # Generate a task ID for each halo
        nx = np.amax(ipos[:,0]) + 1
        ny = np.amax(ipos[:,1]) + 1
        nz = np.amax(ipos[:,2]) + 1
        task_id = ipos[:,2] * nx * ny + ipos[:,1] * nx + ipos[:,0] 

        # Sort halos by task ID
        idx = np.argsort(task_id)
        centre  = so_cat.centre[idx,:]
        index   = so_cat.index[idx]
        radius  = so_cat.r_size[idx]
        task_id = task_id[idx]

        # Find groups of halos with the same task ID
        unique_ids, offsets, counts = np.unique(task_id, return_index=True, return_counts=True)
        
        # Create the task list
        tasks = []
        for offset, count in zip(offsets, counts):
            tasks.append(ChunkTask(index[offset:offset+count], centre[offset:offset+count,:],
                                   radius[offset:offset+count], search_radius, halo_prop_list))

        # Use number of halos as a rough estimate of cost.
        # Do tasks with the most halos first so we're not waiting for a few big jobs at the end.
        tasks.sort(key = lambda x: -x.centres.shape[0])
        self.tasks = tasks

        
class ChunkTask:
    """
    Each ChunkTask is a set of halos in a patch of the simulation volume
    for which we want to evaluate spherical overdensity properties.

    Each ChunkTask is called collectively on all of the MPI ranks in one
    compute node.

    centres contains the halo centres
    search_radius is the radius around each halo we need to read in
    indexes contains the index of each halo in the input catalogue
    """
    def __init__(self, indexes, centres, radii, search_radius,
                 halo_prop_list):

        self.indexes = indexes.copy()
        self.centres = centres.copy()
        self.radii   = radii.copy()
        self.search_radius = search_radius
        self.halo_prop_list = halo_prop_list
        
    def __call__(self, cellgrid, comm, inter_node_rank):

        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()

        def message(m):
            if inter_node_rank >= 0:
                print("Node %d: %s" % (inter_node_rank, m))

        # Find the region we need to read in
        pos_min = np.amin(self.centres, axis=0) - self.search_radius
        pos_max = np.amax(self.centres, axis=0) + self.search_radius

        message("chunk has pos_min=(%.2f,%.2f,%.2f), pos_max=(%.2f,%.2f,%.2f)" %
                (pos_min[0].value, pos_min.value[1],
                 pos_min.value[2], pos_max.value[0], pos_max.value[1],
                 pos_max.value[2]))

        # Find the halo centres, radii etc
        centres = self.centres
        radii   = self.radii
        indexes = self.indexes
        halo_prop_list = self.halo_prop_list

        # Get the cosmology from the input snapshot
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
        data = cellgrid.read_masked_cells_to_shared_memory(properties, mask, comm)

        # Count how many particles we read in
        nr_parts = 0
        for ptype in data:
            name = mass_dataset(ptype)
            nr_parts += data[ptype][name].full.shape[0]
        if nr_parts == 0:
            # Should be impossible: all halos have particles!
            raise Exception("Task has zero particles?!")

        message("read in %d particles" % nr_parts)

        # Do periodic shift of particles to copies nearest the reference point
        for ptype in data:
            if "Coordinates" in data[ptype]:
                data[ptype]["Coordinates"].local[:] = box_wrap(data[ptype]["Coordinates"].local[:], ref_pos, boxsize)

        # Build the mesh for each particle type
        mesh = {}
        for ptype in properties:
            # Find the particle coordinates
            pos = data[ptype]["Coordinates"]
            nr_parts_type = pos.full.shape[0]
            # Compute mesh resolution to give ~1000 particles per cell
            target_nr_per_cell = 1000
            max_resolution = 256
            resolution = int((nr_parts_type/target_nr_per_cell)**(1./3.))
            resolution = min(max(resolution, 1), max_resolution)
            # Build the mesh for this particle type
            mesh[ptype] = shared_mesh.SharedMesh(comm, pos, resolution)

        # Make a list of halo tasks to process, in descending order of radius
        if comm_rank == 0:
            order = np.argsort(-radii)
            tasks = []
            for i in range(len(order)):
                j = order[i]
                tasks.append(halo_tasks.HaloTask(indexes[j], centres[j,:], radii[j]))
        else:
            tasks = None

        message("start %d halo tasks on %d MPI ranks" % (len(tasks), comm_size))

        # Execute the tasks
        results = task_queue.execute_tasks(tasks,
                                           args=(mesh, data, halo_prop_list, a, z, cosmo),
                                           comm_master=comm, comm_workers=MPI.COMM_SELF)

        message("halo tasks done")

        # Combine task results into arrays:
        # Each MPI rank will have a dict of arrays with the results for the halos
        # it processed.
        nr_halos = len(results)
        result_arrays = {}
        for halo_nr, result in enumerate(results):
            for name, (value, description) in result.items():
                if name not in result_arrays:
                    arr = astropy.units.Quantity(-np.ones(nr_halos, dtype=float), unit=value.unit)
                    result_arrays[name] = (arr, description)
                result_arrays[name][0][halo_nr] = value

        message("returning results")

        # Return the results
        return result_arrays
