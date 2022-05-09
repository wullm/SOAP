#!/bin/env python

import unyt
import numpy as np

import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort

import create_groups
import swift_units


def concatenate(result_sets):
    """
    Concatenate a list of result sets into a single set.
    Also clears the input result sets as a side effect.
    """

    # Create the output result set
    output = ResultSet()

    # Ensure input sets are allocated to exactly the right size
    for rs in result_sets:
        rs.trim()

    # Make sure all result sets contains the same quantity names
    for rs in result_sets:
        if sorted(list(rs.result_arrays.keys())) != sorted(list(result_sets[0].result_arrays.keys())):
            raise Exception("Attempting to concatenate inconsistent ResultSets!")
    names = list(result_sets[0].result_arrays.keys())

    # Check units, dtype and shape for all quantities
    for name in names:
        arr1 = rs.result_arrays[name][0]
        arr2 = result_sets[0].result_arrays[name][0]
        if arr1.dtype != arr2.dtype:
            raise Exception(f"Result arrays for quantity {name} have inconsistent dtypes!")
        if arr1.units != arr2.units:
            raise Exception(f"Result arrays for quantity {name} have inconsistent units!")
        if arr1.shape[1:] != arr2.shape[1:]:
            raise Exception(f"Result arrays for quantity {name} have inconsistent shapes!")

    # Compute total number of halos in the output
    output.nr_halos = sum([rs.nr_halos for rs in result_sets])

    # Concatenate arrays
    for name in names:

        # Store concatenated arrays in the output
        list_of_arrays = [rs.result_arrays[name][0] for rs in result_sets]
        concatenated_array = unyt.array.uconcatenate(list_of_arrays, axis=0)
        description  = rs.result_arrays[name][1]
        output.result_arrays[name] = [concatenated_array, description]

        # Free input arrays (to avoid storing two copies of the full dataset at once)
        for rs in result_sets:
            del rs.result_arrays[name]
            rs.nr_halos = 0

    return output


class ResultSet:
    """
    Class to store halo properties as a dict of (unyt_array, description)
    tuples.
    """
    def __init__(self):
        
        # Initially we have no data arrays
        self.nr_halos = 0
        self.initial_size = 1000
        self.result_arrays = {}

    def append(self, results):
        """
        Append results for a halo

        results - dict of the form results[name] = (description, data)
        where name is the name of the quantity, description is a description
        of the quantity, and data is a unyt_array with the value.
        """
    
        # Loop over quantities to store
        for result_name, (result_data, result_description) in results.items():

            # Create a new array to store this property if necessary
            if result_name not in self.result_arrays:
                shape = (self.initial_size,) + result_data.shape
                # need to ensure we don't pass a unyt_quantity to empty_like
                # because unyt_quantities must be scalars only
                arr = np.empty_like(unyt.unyt_array(result_data, registry=result_data.units.registry), shape=shape)
                self.result_arrays[result_name] = [arr, result_description]

            # Find the array to store this result
            result_array, result_description = self.result_arrays[result_name]

            # Consistency check: data type, units and shape should match the existing array
            if result_data.units != result_array.units:
                raise Exception(f"Result units are inconsistent for quantity {result_name}")
            if result_data.dtype != result_array.dtype:
                raise Exception(f"Result dtypes are inconsistent for quantity {result_name}")
            if result_data.shape != result_array.shape[1:]:
                raise Exception(f"Result shapes are inconsistent for quantity {result_name}")                            

            # Ensure the array is large enough
            if self.nr_halos >= result_array.shape[0]:
                new_shape = list(result_array.shape)
                new_shape[0] *= 2
                new_result_array = np.empty_like(unyt.unyt_array(result_array, registry=result_data.units.registry), shape=new_shape)
                new_result_array[0:result_array.shape[0],...] = result_array[...]
                result_array = new_result_array
                self.result_arrays[result_name] = [result_array, result_description]

            # Store this property for this halo to the output array
            result_array[self.nr_halos,...] = result_data

        # Increment number of halos stored
        self.nr_halos += 1

    def trim(self):
        """
        Trim arrays down to exactly the required size
        """
        for name in self.result_arrays:
            self.result_arrays[name][0] = self.result_arrays[name][0][:self.nr_halos,...]

    def parallel_sort(self, key, comm):
        """
        Do an MPI parallel sort of the results on the specified sort key
        """
        
        # First, make sure everyone has the same set of array names
        names_local = list(self.result_arrays.keys())
        names_ref = comm.bcast(names_local)
        if names_local != names_ref:
            raise Exception("Names of result arrays are not consistent between MPI ranks!")
        
        # Make sure units, dtype and shapes also agree between MPI ranks
        for name in names_ref:
            dtype_local = self.result_arrays[name][0].dtype
            units_local = self.result_arrays[name][0].units
            shape_local = self.result_arrays[name][0].shape[1:]
            dtype_ref, units_ref, shape_ref = comm.bcast((dtype_local, units_local, shape_local))
            if dtype_local != dtype_ref:
                raise Exception(f"Results for {name} have inconsistent dtype between MPI ranks!")
            if units_local != units_ref:
                raise Exception(f"Results for {name} have inconsistent units between MPI ranks!")
            if shape_local != shape_ref:
                raise Exception(f"Results for {name} have inconsistent shape between MPI ranks!")

        # Now make a sorting index on the specified key
        idx = psort.parallel_sort(self.result_arrays[key][0], comm=comm, return_index=True)

        # And reorder the other arrays
        for name in names_ref:
            if name != key:
                self.result_arrays[name][0] = psort.fetch_elements(self.result_arrays[name][0], idx, comm=comm)
        del idx

    def collective_write(self, outfile, comm):
        """
        Write the results to a file in collective mode
        """

        # Get names in a consistent order between MPI ranks
        names = comm.bcast(list(self.result_arrays.keys()))

        # Ensure any HDF5 groups we need exist
        group_names = comm.bcast(create_groups.find_groups_to_create(names))
        for group_name in group_names:
            outfile.create_group(group_name)

        # Loop over output arrays
        for name in names:

            # Write this array
            data, description = self.result_arrays[name]
            phdf5.collective_write(outfile, name, data, comm)

            # Attach units metadata and description
            if hasattr(data, "units"):
                attrs = swift_units.attributes_from_units(data.units)
                for attr_name, attr_value in attrs.items():
                    outfile[name].attrs[attr_name] = attr_value
            outfile[name].attrs["Description"] = description

