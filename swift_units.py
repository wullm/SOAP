#!/bin/env python

import unyt
import unyt.dimensions as dim
import numpy
import h5py


def unit_registry_from_snapshot(snap, name="Units"):
    """
    Create a new unit registry from a SWIFT snapsnot, using the exact
    units and physical constants from SWIFT.

    snap: the snapshot file as a h5py.File object

    Returns a unyt UnitRegistry.
    """

    # Read snapshot metadata
    physical_constants_cgs = {name : float(value) for name, value in snap["PhysicalConstants/CGS"].attrs.items()}
    snap_units_cgs = {name : float(value) for name, value in snap[name].attrs.items()}
    cosmology = {name : float(value) for name, value in snap["Cosmology"].attrs.items()}
    a = cosmology["Scale-factor"]
    h = cosmology["h"]

    # Create a new unit system corresponding to the snapshot units
    us = unyt.UnitSystem(
        snap.filename,
        unyt.unyt_quantity(snap_units_cgs["Unit length in cgs (U_L)"],      units=unyt.cm),
        unyt.unyt_quantity(snap_units_cgs["Unit mass in cgs (U_M)"],        units=unyt.g),
        unyt.unyt_quantity(snap_units_cgs["Unit time in cgs (U_t)"],        units=unyt.s),
        unyt.unyt_quantity(snap_units_cgs["Unit temperature in cgs (U_T)"], unyt.K),
        unyt.rad,
        unyt.unyt_quantity(snap_units_cgs["Unit current in cgs (U_I)"],     units=unyt.A),
    )

    # Create a new unit registry
    reg = unyt.UnitRegistry(unit_system=us)

    # Add the expansion factor as a dimensionless "unit"
    reg.add("a", a, dim.dimensionless)
    reg.add("h", h, dim.dimensionless)

    # Add physical length units
    parsec_cgs = physical_constants_cgs["parsec"]
    parsec_si = parsec_cgs/100.0
    reg.add("pMpc", parsec_si*1.0e6, dim.length)

    # Add mass units
    solar_mass_cgs = physical_constants_cgs["solar_mass"]
    solar_mass_si = solar_mass_cgs / 1000.0
    reg.add("Msun", solar_mass_si, dim.mass)

    return reg


def units_from_attributes(attrs, registry):
    """
    Create a unyt.Unit object from dataset attributes

    attrs: the SWIFT dataset attributes dict
    registry: unyt unit registry with a, h and unit system for the snapshot

    Returns a unyt Unit object.
    """
    # Determine unyt unit for this quantity
    u = unyt.dimensionless
    unit_system = registry.unit_system
    base = registry.unit_system.base_units
    for symbol, baseunit in (("I", base[dim.current_mks]),
                             ("L", base[dim.length]),
                             ("M", base[dim.mass]),
                             ("T", base[dim.temperature]),
                             ("t", base[dim.time])):
        unit = unyt.Unit(baseunit, registry=registry)
        exponent = attrs["U_%s exponent" % symbol][0]
        if exponent == 1.0:
            if u is unyt.dimensionless:
                u = unit
            else:
                u = u*unit
        elif exponent != 0.0:
            if u is unyt.dimensionless:
                u = unit**exponent
            else:
                u = u*(unit**exponent)

    # Add expansion factor
    a_scale_exponent = attrs["a-scale exponent"][0]
    a_unit = unyt.Unit("a", registry=registry)**a_scale_exponent
    if a_scale_exponent != 0:
        if u is unyt.dimensionless:
            u = a_unit
        else:
            u = u*a_unit

    # Add h factor
    h_scale_exponent = attrs["h-scale exponent"][0]
    h_unit = unyt.Unit("h", registry=registry)**h_scale_exponent
    if h_scale_exponent != 0:
        if u is unyt.dimensionless:
            u = h_unit
        else:
            u = u*h_unit

    return unyt.Unit(u, registry=registry)


def attributes_from_units(units):
    """
    Given a unyt.Unit object, generate SWIFT dataset attributes

    units: the Unit object

    Returns a dict with the attributes
    """
    attrs = {}

    # Get CGS conversion factor. Note that this is the conversion to physical units,
    # because unyt multiplies out the dimensionless a factor.
    cgs_factor, offset = units.get_conversion_factor(units.get_cgs_equivalent())

    # Get a exponent
    a_unit = unyt.Unit("a", registry=units.registry)
    a_exponent = units.expr.as_powers_dict()[a_unit.expr]
    a_val = a_unit.base_value

    # Get h exponent
    h_unit = unyt.Unit("h", registry=units.registry)
    h_exponent = units.expr.as_powers_dict()[h_unit.expr]
    h_val = h_unit.base_value

    # Find the power associated with each dimension
    powers = units.get_mks_equivalent().dimensions.as_powers_dict()
    
    # Set the attributes
    attrs["Conversion factor to CGS (not including cosmological corrections)"] = [float(cgs_factor/(a_val**a_exponent)/(h_val**h_exponent)),]
    attrs["Conversion factor to CGS (including cosmological corrections)"]     = [float(cgs_factor),]
    attrs["U_I exponent"] = [float(powers[unyt.dimensions.current_mks]),]
    attrs["U_L exponent"] = [float(powers[unyt.dimensions.length]),]
    attrs["U_M exponent"] = [float(powers[unyt.dimensions.mass]),]
    attrs["U_T exponent"] = [float(powers[unyt.dimensions.temperature]),]
    attrs["U_t exponent"] = [float(powers[unyt.dimensions.time]),]
    attrs["a-scale exponent"] = [float(a_exponent),]
    attrs["h-scale exponent"] = [float(h_exponent),]

    return attrs
