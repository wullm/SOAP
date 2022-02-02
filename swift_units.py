#!/bin/env python

import astropy.units as u
import collections

def units_from_attributes(dset):
    cgs_factor = dset.attrs["Conversion factor to CGS (not including cosmological corrections)"][0]
    U_I = dset.attrs["U_I exponent"][0]
    U_L = dset.attrs["U_L exponent"][0]
    U_M = dset.attrs["U_M exponent"][0]
    U_T = dset.attrs["U_T exponent"][0]
    U_t = dset.attrs["U_t exponent"][0]
    return cgs_factor * (u.A**U_I) * (u.cm**U_L) * (u.g**U_M) * (u.K**U_T) * (u.s**U_t) 

def write_unit_attributes(dset, unit):
    cgs_factor = unit.cgs.scale
    powers = collections.defaultdict(lambda: 0)
    for base, power in zip(unit.cgs.bases, unit.cgs.powers):
        if base == u.cm:
            powers["U_L"] = power
        elif base == u.g:
            powers["U_M"] = power
        elif base == u.A:
            powers["U_I"] = power
        elif base == u.K:
            powers["U_T"] = power
        elif base == u.s:
            powers["U_t"] = power
    dset.attrs["Conversion factor to CGS (not including cosmological corrections)"] = [cgs_factor,]
    dset.attrs["U_I exponent"] = [powers["U_I"],]
    dset.attrs["U_L exponent"] = [powers["U_L"],]
    dset.attrs["U_M exponent"] = [powers["U_M"],]
    dset.attrs["U_T exponent"] = [powers["U_T"],]
    dset.attrs["U_t exponent"] = [powers["U_t"],]
    dset.attrs["Astropy unit string representation"] = unit.to_string()
