#!/bin/env python

import numpy as np
import halo_particles

class SOTaskList:
    """
    Stores a list of SOTasks to be executed.
    """
    def __init__(self, cellgrid, so_cat, search_radius, cells_per_task):
                
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
            tasks.append(SOTask(index[offset:offset+count], centre[offset:offset+count,:],
                                radius[offset:offset+count], search_radius))

        # Use number of halos as a rough estimate of cost.
        # Do tasks with the most halos first so we're not waiting for a few big jobs at the end.
        tasks.sort(key = lambda x: -x.centres.shape[0])
        self.tasks = tasks
        
class SOTask:
    """
    Each SOTask is a set of halos in a patch of the simulation volume
    for which we want to evaluate spherical overdensity properties.

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
        
    def bounding_box(self):
        pos_min = np.amin(self.centres, axis=0) - self.search_radius
        pos_max = np.amax(self.centres, axis=0) + self.search_radius
        return pos_min, pos_max

    def volume(self):
        pos_min, pos_max = self.bounding_box()
        r = pos_max - pos_min
        return r[0]*r[1]*r[2]

    def run(self, cellgrid):
        pos_min, pos_max = self.bounding_box()
        result = halo_particles.compute_so_properties(cellgrid, self.centres, self.radii, pos_min, pos_max,
                                                      halo_prop_list)

        # Add an extra result array with the original index of the halo
        result["index"] = (self.indexes, "Position of the halo in the VR catalogue")

        return result
