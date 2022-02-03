#!/bin/env python

def mass_dataset(ptype):
    if ptype == "PartType5":
        return "DynamicalMasses"
    else:
        return "Masses"
