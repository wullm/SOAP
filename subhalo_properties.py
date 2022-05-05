#!/bin/env python

import numpy as np
import unyt

from halo_properties import HaloProperty
from dataset_names import mass_dataset


class SubhaloMasses(HaloProperty):

    def __init__(self, cellgrid, bound_only=True):
        super().__init__(cellgrid)

        self.bound_only = bound_only

        # This specifies how large a sphere is read in:
        self.mean_density_multiple = None
        self.critical_density_multiple = None

        # Minimum physical radius to read in (pMpc)
        self.physical_radius_mpc = 0.0

        # Give this calculation a name so we can select it on the command line
        if bound_only:
            self.grnr = "GroupNr_bound"
            self.name = "subhalo_masses_bound"
        else:
            self.grnr = "GroupNr_all"
            self.name = "subhalo_masses_all"            

        # Arrays which must be read in for this calculation.
        # Note that if there are no particles of a given type in the
        # snapshot, that type will not be read in and will not have
        # an entry in the data argument to calculate(), below.
        # (E.g. gas, star or BH particles in DMO runs)
        self.particle_properties = {
            "PartType0" : ["Coordinates", "Velocities", "Masses", self.grnr],
            "PartType1" : ["Coordinates", "Velocities", "Masses", self.grnr],
            "PartType4" : ["Coordinates", "Velocities", "Masses", "InitialMasses", self.grnr],
            "PartType5" : ["Coordinates", "Velocities", "DynamicalMasses", "SubgridMasses", self.grnr]
        }

    def calculate(self, input_halo, data, halo_result):
        """
        Compute centre of mass etc of bound particles

        input_halo       - dict with halo properties passed in from VR (see
                           halo_centres.py)
        data             - contains particle data. E.g. data["PartType1"]["Coordinates"]
                           has the particle coordinates for type 1
        halo_result      - dict with halo properties computed so far. Properties
                           computed here should be added to halo_result.

        Input particle data arrays are unyt_arrays.
        """
        
        # Storage for properties of each particle type
        cofm_pos   = {}
        cofm_vel   = {}
        total_mass = {}
        nr_part    = {}

        # Look up array index of this halo in VR catalogue
        index = input_halo["index"]

        # Find the simulation mass unit
        mass_unit = unyt.Unit("snap_mass",   registry=self.unit_registry)
        zero_mass = unyt.unyt_quantity(0.0, units=mass_unit)

        # Loop over particle types
        total_initial_mass = zero_mass.copy()
        total_subgrid_mass = zero_mass.copy()
        pos_times_mass = None
        pos_units = None
        pos_dtype = None
        vel_times_mass = None
        vel_units = None
        vel_dtype = None
        for ptype in data:

            # Find position and mass of particles in the group
            grnr = data[ptype][self.grnr]
            in_halo = (grnr==index)
            pos  = data[ptype]["Coordinates"][in_halo,:]
            vel  = data[ptype]["Velocities"][in_halo,:]
            mass = data[ptype][mass_dataset(ptype)][in_halo]

            # Record the position and velocity units
            if pos_units is None:
                pos_units = pos.units
                pos_dtype = pos.dtype
                vel_units = vel.units
                vel_dtype = vel.dtype

            # Store total mass of particles of this type
            total_mass[ptype] = np.sum(mass, dtype=float)
        
            # Accumulate total number of particles of this type
            nr_part[ptype] = unyt.unyt_quantity(pos.shape[0], units=unyt.dimensionless, dtype=int)

            # Total stellar initial mass
            if ptype == "PartType4":
                total_initial_mass = np.sum(data[ptype]["InitialMasses"][in_halo], dtype=float)

            # Total BH subgrid mass
            if ptype == "PartType5":
                total_subgrid_mass = np.sum(data[ptype]["SubgridMasses"][in_halo], dtype=float)

            # Accumulate pos*mass for centre of mass
            if pos_times_mass is None:
                pos_times_mass = (pos*mass[:, None]).sum(axis=0)
            else:
                pos_times_mass += (pos*mass[:, None]).sum(axis=0)

            # Accumulate vel*mass for centre of mass velocity
            if vel_times_mass is None:
                vel_times_mass = (vel*mass[:, None]).sum(axis=0)
            else:
                vel_times_mass += (vel*mass[:, None]).sum(axis=0)

        # Find total masses
        total_mass_all = np.sum(unyt.unyt_array([total_mass[ptype] for ptype in data]))
        nr_part_all    = np.sum(unyt.unyt_array([nr_part[ptype] for ptype in data]))

        # Compute centre of mass
        if pos_times_mass is not None:
            cofm_pos = (pos_times_mass / total_mass_all) % self.boxsize
        else:
            cofm_pos = unyt.unyt_array((0,0,0), units=pos_units, dtype=pos_dtype)

        # Compute centre of mass velocity
        if vel_times_mass is not None:
            cofm_vel = vel_times_mass / total_mass_all
        else:
            cofm_vel = unyt.unyt_array((0,0,0), units=vel_units, dtype=vel_dtype)

        # Add these properties to the output
        if self.bound_only:
            prefix="BoundSubhaloParticles/"
            label="which are bound to the halo"
        else:
            prefix="AllSubhaloParticles"
            label="which are bound or unbound"
        for ptype in nr_part:
            halo_result[f"{prefix}/NumPart_{ptype}"]              = (nr_part[ptype],     f"Number of particles of type {ptype} {label}")
            halo_result[f"{prefix}/Mass_{ptype}"]                 = (total_mass[ptype],  f"Total mass of particles of type {ptype} {label}")
        halo_result[f"{prefix}/StellarInitialMass"]               = (total_initial_mass, "Total initial mass of star particles {label}")
        halo_result[f"{prefix}/BHSubgridMass"]                    = (total_subgrid_mass, "Total subgrid mass of black hole particles {label}")
        halo_result[f"{prefix}/Mass_All"]                         = (total_mass_all,     "Total mass of all particle types (excluding neutrinos) {label}")
        halo_result[f"{prefix}/NumPart_All"]                      = (nr_part_all,        "Total number of particles of all types (excluding neutrinos) {label}")
        halo_result[f"{prefix}/CentreOfMass"]                     = (cofm_pos,           "Centre of mass position {label}")
        halo_result[f"{prefix}/CentreOfMassVelocity"]             = (cofm_vel,           "Centre of mass velocity {label}")

