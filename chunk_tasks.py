#!/bin/env python

import time
import numpy as np
import astropy.units
from mpi4py import MPI

import task_queue
import shared_mesh
import shared_array
import halo_tasks
from dataset_names import mass_dataset
from halo_tasks import process_halos


def box_wrap(pos, ref_pos, boxsize):
    shift = ref_pos[None,:] - 0.5*boxsize
    return (pos - shift) % boxsize + shift

time_start = time.time()

class ChunkTaskList:
    """
    Stores a list of ChunkTasks to be executed.
    """
    def __init__(self, cellgrid, so_cat, search_radius, chunks_per_dimension,
                 halo_prop_list):

        # Determine size of a task. Must be at least one top level cell.
        cells_per_task = max(1, cellgrid.dimension[0] // chunks_per_dimension)

        # Find size of volume associated with each task
        task_size = cellgrid.cell_size[0]*cells_per_task
        
        # For each centre, determine integer coords in task grid
        ipos = np.floor(so_cat.centre / task_size).to(1).value.astype(int)

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
    def __init__(self, indexes=None, centres=None, radii=None,
                 search_radius=None, halo_prop_list=None):

        self.indexes = indexes
        self.centres = centres
        self.radii = radii
        self.search_radius = search_radius
        self.halo_prop_list = halo_prop_list
        self.shared = False
        
    def __call__(self, cellgrid, comm, inter_node_rank):

        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()
        
        def message(m):
            if inter_node_rank >= 0:
                print("[%8.1fs] %d: %s" % (time.time()-time_start, inter_node_rank, m))

        # Find the region we need to read in
        if comm_rank == 0:
            pos_min = np.amin(self.centres.full, axis=0) - self.search_radius
            pos_max = np.amax(self.centres.full, axis=0) + self.search_radius
        else:
            pos_min = None
            pos_max = None
        pos_min, pos_max = comm.bcast((pos_min, pos_max))

        message("chunk has pos_min=(%.2f,%.2f,%.2f), pos_max=(%.2f,%.2f,%.2f)" %
                (pos_min[0].value, pos_min.value[1], pos_min.value[2],
                 pos_max.value[0], pos_max.value[1], pos_max.value[2]))

        # Get the cosmology from the input snapshot
        cosmo = cellgrid.cosmology
        a = cellgrid.a
        z = cellgrid.z
        boxsize = cellgrid.boxsize
        ref_pos = (pos_min+pos_max)/2

        # Find all particle properties we need to read in:
        # For each particle type this is the union of the quantities
        # needed for each calculation.
        if comm_rank == 0:
            properties = {}
            for halo_prop in self.halo_prop_list:
                for ptype in halo_prop.particle_properties:
                    if ptype not in properties:
                        properties[ptype] = set()
                    properties[ptype] = properties[ptype].union(halo_prop.particle_properties[ptype])
            for ptype in properties:
                properties[ptype] = list(properties[ptype])
        else:
            properties = None
        properties = comm.bcast(properties)

        # Don't try to read particle types which don't exist in the snapshot
        for ptype in list(properties.keys()):
            if ptype not in cellgrid.ptypes:
                del properties[ptype]

        # Read in particles in the required region
        comm.barrier()
        t0_read = time.time()
        mask = cellgrid.empty_mask()
        cellgrid.mask_region(mask, pos_min, pos_max)
        data = cellgrid.read_masked_cells_to_shared_memory(properties, mask, comm)
        comm.barrier()
        t1_read = time.time()

        # Count how many particles we read in
        nr_parts = 0
        for ptype in data:
            name = mass_dataset(ptype)
            nr_parts += data[ptype][name].full.shape[0]
        if nr_parts == 0:
            # Should be impossible: all halos have particles!
            raise Exception("Task has zero particles?!")

        # Compute number of bytes read
        nr_bytes = 0
        for ptype in data:
            for name in data[ptype]:
                nr_bytes += data[ptype][name].full.nbytes
        nr_mb = nr_bytes/(1024**2)
        rate = nr_mb/(t1_read-t0_read)
        message("read in %d particles in %.1fs = %.1fMB/s (uncompressed)" % (nr_parts, t1_read-t0_read, rate))

        # Do periodic shift of particles to copies nearest the reference point
        for ptype in data:
            if "Coordinates" in data[ptype]:
                data[ptype]["Coordinates"].local[:] = box_wrap(data[ptype]["Coordinates"].local[:], ref_pos, boxsize)

        # Build the mesh for each particle type
        comm.barrier()
        t0_mesh = time.time()
        mesh = {}
        for ptype in properties:
            # Find the particle coordinates
            pos = data[ptype]["Coordinates"]
            nr_parts_type = pos.full.shape[0]
            # Compute mesh resolution to give roughly fixed number of particles per cell
            target_nr_per_cell = 1000
            max_resolution = 256
            resolution = int((nr_parts_type/target_nr_per_cell)**(1./3.))
            resolution = min(max(resolution, 1), max_resolution)
            # Build the mesh for this particle type
            mesh[ptype] = shared_mesh.SharedMesh(comm, pos, resolution)
        comm.barrier()
        t1_mesh = time.time()
        message("constructing shared mesh took %.1fs" % (t1_mesh-t0_mesh))

        # Calculate the halo properties
        t0_halos = time.time()
        nr_halos = len(self.indexes.full)
        result, dead_time_fraction = process_halos(comm, data, mesh, self.halo_prop_list, a, z, cosmo,
                                                   boxsize, self.indexes, self.centres, self.radii)
        t1_halos = time.time()
        message("processing %d halos on %d ranks took %.1fs (dead time frac.=%.2f)" % (nr_halos, comm_size,
                                                                                       t1_halos-t0_halos, dead_time_fraction))

        # Free the shared particle data
        for ptype in data:
            for name in data[ptype]:
                data[ptype][name].free()

        # Free the shared mesh
        for ptype in mesh:
            mesh[ptype].free()

        # Free shared halo catalogue
        if self.shared:
            self.indexes.free()
            self.centres.free()
            self.radii.free()

        return result

    @classmethod
    def bcast(cls, comm, instance):
        """
        Broadcast a class instance over communicator comm.
        instance is only significant on rank 0. Other ranks
        don't have an instance yet, so this is a class method.
        """

        comm_rank = comm.Get_rank()

        # Create a class instance on ranks which don't have one
        if comm_rank == 0:
            self = instance
        else:
            self = cls()

        # Share the data arrays
        def share_array(arr):
            if comm_rank == 0:
                shape = list(arr.shape)
                dtype = arr.dtype
                if hasattr(arr, "unit"):
                    unit = arr.unit
                else:
                    unit = None
            else:
                shape = None
                dtype = None
                unit = None
            shape, dtype, unit = comm.bcast((shape, dtype, unit))
            if comm_rank > 0:
                shape[0] = 0
            shared_arr = shared_array.SharedArray(shape, dtype, comm, unit)
            if comm_rank == 0:
                shared_arr.full[...] = arr[...]
            shared_arr.sync()
            return shared_arr

        self.indexes = share_array(self.indexes)
        self.centres = share_array(self.centres)
        self.radii   = share_array(self.radii)
        self.search_radius = comm.bcast(self.search_radius)
        self.halo_prop_list = comm.bcast(self.halo_prop_list)

        self.shared = True

        return self
