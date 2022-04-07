#!/bin/env python

import numpy as np
import unyt
import swiftsimio.objects as o

def cosmo_array_like(arr, dtype=None, shape=None, comoving=None, cosmo_factor=None, units=None):
    """
    Make a new cosmo_array like the input array, possibly overriding some attributes
    """
    
    dtype = arr.dtype if dtype is None else dtype
    shape = arr.shape if shape is None else shape
    units = arr.units if units is None else units
    comoving = arr.comoving if comoving is None else comoving
    cosmo_factor = arr.cosmo_factor if cosmo_factor is None else cosmo_factor
    
    data = np.ndarray(shape, dtype=dtype)
    data = unyt.unyt_array(data, units=units)
    data = o.cosmo_array(data, comoving=comoving, cosmo_factor=cosmo_factor)
    return data

def cosmo_array_scalar(value, dtype, a, units=None, a_exponent=0.0, comoving=False):
    """
    Create a new cosmo_array scalar with the specified value and parameters
    """
    data = np.ndarray((), dtype=dtype)
    data[()] = value
    data = unyt.unyt_array(data, units=units)
    cosmo_factor = o.cosmo_factor(o.a**a_exponent, a)
    data = o.cosmo_array(data, comoving=comoving, cosmo_factor=cosmo_factor)
    return data

def cosmo_array_zeros(shape, dtype, a, units=None, a_exponent=0.0, comoving=False):
    """
    Create a new cosmo_array vector zero with the specified parameters
    """
    data = np.zeros(shape, dtype=dtype)
    data = unyt.unyt_array(data, units=units)
    cosmo_factor = o.cosmo_factor(o.a**a_exponent, a)
    data = o.cosmo_array(data, comoving=comoving, cosmo_factor=cosmo_factor)
    return data
