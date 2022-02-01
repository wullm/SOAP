#!/bin/env python

import numpy as np

class SOTaskList:
    """
    Stores a list of SOTasks to be executed.
    """
    def __init__(self, cellgrid, so_cat, search_radius, cells_per_task):
                
        # For each centre, determine integer coords in task grid
        ipos = np.floor(so_cat.centre / cellgrid.cell_size[None,:]).astype(int) // cells_per_task

        # Generate a task ID for each halo
        nx = np.amax(ipos[:,0])
        ny = np.amax(ipos[:,1])
        nz = np.amax(ipos[:,2])
        task_id = ipos[:,2] * nx * ny + ipos[:,1] * nx + ipos[:,0] 

        # Sort halos by task ID
        idx = np.argsort(task_id)
        centre  = so_cat.centre[idx,:]
        index   = so_cat.index[idx]
        task_id = task_id[idx]

        # Find groups of halos with the same task ID
        unique_ids, offsets, counts = np.unique(task_id, return_index=True, return_counts=True)
        
        # Create the task list
        tasks = []
        for offset, count in zip(offsets, counts):
            tasks.append(SOTask(index[offset:offset+count], centre[offset:offset+count,:], search_radius))

        self.tasks = tasks

class SOTask:
    """
    Each SOTask is a set of halos in a patch of the simulation volume
    for which we want to evaluate spherical overdensity properties.

    centres contains the halo centres
    search_radius is the radius around each halo we need to read in
    indexes contains the index of each halo in the input catalogue
    """
    def __init__(self, indexes, centres, search_radius):
        self.indexes = indexes.copy()
        self.centres = centres.copy()
        self.search_radius = search_radius

    def bounding_box(self):
        pos_min = np.amin(self.centres, axis=0) - self.search_radius
        pos_max = np.amax(self.centres, axis=0) + self.search_radius
        return pos_min, pos_max

    def run(self, cellgrid):

        pos_min, pos_max = self.bounding_box()
        self.result = halo_properties.compute_so_properties(self.centres, pos_min, pos_max)

