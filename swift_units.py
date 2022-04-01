#!/bin/env python

import swiftsimio.objects as o
import unyt

def units_from_attributes(dset, unit_system, a):

    # Determine unyt unit for this quantity
    u = 1.0
    for symbol, unit in (("I", unit_system.current),
                         ("L", unit_system.length),
                         ("M", unit_system.mass),
                         ("T", unit_system.temperature),
                         ("t", unit_system.time)):
        exponent = dset.attrs["U_%s exponent" % symbol][0]
        if exponent == 1.0:
            u *= unit
        elif exponent != 0.0:
            u *= (unit**exponent)

    # Handle any a factor(s) and return a swiftsimio cosmo_array
    a_scale_exponent = dset.attrs["a-scale exponent"][0]
    cosmo_factor = o.cosmo_factor(o.a**a_scale_exponent, a)
    return o.cosmo_array(unyt.unyt_array(1.0, units=u), cosmo_factor=cosmo_factor, comoving=(a_scale_exponent!=0.0))

def write_unit_attributes(dset, cosmo_array):
    
    # Get CGS conversion factor and a exponent from the cosmo_array's units
    u = cosmo_array.units
    cgs_factor = u.units.get_conversion_factor(u.get_cgs_equivalent())
    a_exponent = u.cosmo_factor.expr.exp

    # Find the power associated with each dimension: SWIFT treats current as
    # a base unit. The CGS system in unyt does not, so convert to MKS to get
    # dimensions.
    powers = u.get_mks_equivalent().dimensions.as_powers_dict()
    
    # Write the attributes
    dset.attrs["Conversion factor to CGS (not including cosmological corrections)"] = [cgs_factor,]
    dset.attrs["Conversion factor to CGS (including cosmological corrections)"]     = [cgs_factor*(a**a_exponent),]
    dset.attrs["U_I exponent"] = [powers[unyt.dimensions.current_mks],]
    dset.attrs["U_L exponent"] = [powers[unyt.dimensions.length],]
    dset.attrs["U_M exponent"] = [powers[unyt.dimensions.mass],]
    dset.attrs["U_T exponent"] = [powers[unyt.dimensions.temperature],]
    dset.attrs["U_t exponent"] = [powers[unyt.dimensions.time],]
    dset.attrs["a-scale exponent"] = [float(a_exponent),]
    dset.attrs["h-scale exponent"] = [0.0,]

