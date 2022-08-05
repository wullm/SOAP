#!/bin/env python

from exclusive_sphere_properties import ExclusiveSphereProperties
from projected_aperture_properties import ProjectedApertureProperties
from SO_properties import SOProperties
from subhalo_properties import SubhaloProperties

import unyt


class PropertyTable:
    def __init__(self):
        self.properties = {}

    def add_properties(self, halo_property):
        halo_type = halo_property.__name__
        props = halo_property.property_list
        for prop_name, prop_shape, prop_dtype, prop_units, prop_description in props:
            prop_units = unyt.unyt_quantity(1, units=prop_units).units.latex_repr
            if prop_name in self.properties:
                # run some checks
                if prop_shape != self.properties[prop_name]["shape"]:
                    print("Shape mismatch!")
                    print(halo_type, prop_name, prop_shape, self.properties[prop_name])
                    exit()
                if prop_dtype != self.properties[prop_name]["dtype"]:
                    print("dtype mismatch!")
                    print(halo_type, prop_name, prop_dtype, self.properties[prop_name])
                    exit()
                if prop_units != self.properties[prop_name]["units"]:
                    print("Unit mismatch!")
                    print(halo_type, prop_name, prop_units, self.properties[prop_name])
                    exit()
                if prop_description != self.properties[prop_name]["description"]:
                    print("Description mismatch!")
                    print(
                        halo_type,
                        prop_name,
                        prop_description,
                        self.properties[prop_name],
                    )
                    exit()
            else:
                self.properties[prop_name] = {
                    "shape": prop_shape,
                    "dtype": prop_dtype,
                    "units": prop_units,
                    "description": prop_description,
                    "types": [halo_type],
                }

    def print_table(self):
        names = sorted(list(self.properties.keys()))
        for name in names:
            print(name)


if __name__ == "__main__":

    table = PropertyTable()
    table.add_properties(ExclusiveSphereProperties)
    table.add_properties(ProjectedApertureProperties)
    table.add_properties(SOProperties)
    table.add_properties(SubhaloProperties)

    table.print_table()
