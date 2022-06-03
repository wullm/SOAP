#!/bin/env python

import numpy as np
import unyt

from halo_properties import HaloProperty
from dataset_names import mass_dataset


class ProjectedApertureProperties(HaloProperty):
    def __init__(self, cellgrid, physical_radius_kpc):
        super().__init__(cellgrid)

        # This specifies how large a sphere is read in:
        self.mean_density_multiple = None
        self.critical_density_multiple = None

        # Minimum physical radius to read in (pMpc)
        self.physical_radius_mpc = 0.001 * physical_radius_kpc

        self.name = f"projected_aperture_{physical_radius_kpc:.0f}kpc"

        self.particle_properties = {
            "PartType0": ["Coordinates", "Velocities", "Masses", "GroupNr_all"],
            "PartType1": ["Coordinates", "Velocities", "Masses", "GroupNr_all"],
            "PartType4": [
                "Coordinates",
                "Velocities",
                "Masses",
                "InitialMasses",
                "GroupNr_all",
            ],
            "PartType5": [
                "Coordinates",
                "Velocities",
                "DynamicalMasses",
                "SubgridMasses",
                "GroupNr_all",
            ],
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

        centre = input_halo["cofp"]
        index = input_halo["index"]

        types_present = [type for type in self.particle_properties if type in data]

        mass = []
        position = []
        radius_projx = []
        radius_projy = []
        radius_projz = []
        velocity = []
        types = []
        for ptype in types_present:
            grnr = data[ptype]["GroupNr_all"]
            in_halo = grnr == index
            mass.append(data[ptype][mass_dataset(ptype)][in_halo])
            pos = data[ptype]["Coordinates"][in_halo, :] - centre[None, :]
            position.append(pos)
            rprojx = np.sqrt(pos[:, 1] ** 2 + pos[:, 2] ** 2)
            radius_projx.append(rprojx)
            rprojy = np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)
            radius_projy.append(rprojy)
            rprojz = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
            radius_projz.append(rprojz)
            velocity.append(data[ptype]["Velocities"][in_halo, :])
            typearr = np.zeros(rprojx.shape, dtype="U9")
            typearr[:] = ptype
            types.append(typearr)

        mass = unyt.array.uconcatenate(mass)
        position = unyt.array.uconcatenate(position)
        radius_projx = unyt.array.uconcatenate(radius_projx)
        radius_projy = unyt.array.uconcatenate(radius_projy)
        radius_projz = unyt.array.uconcatenate(radius_projz)
        velocity = unyt.array.uconcatenate(velocity)
        types = np.concatenate(types)

        mask_projx = radius_projx <= self.physical_radius_mpc * unyt.Mpc
        mask_projy = radius_projy <= self.physical_radius_mpc * unyt.Mpc
        mask_projz = radius_projz <= self.physical_radius_mpc * unyt.Mpc

        for projname, projmask in zip(
            ["projx", "projy", "projz"], [mask_projx, mask_projy, mask_projz]
        ):
            proj_mass = mass[projmask]
            proj_position = position[projmask]
            proj_velocity = velocity[projmask]

            proj_Mtot = proj_mass.sum()

            proj_com = (proj_mass[:, None] * proj_position).sum(axis=0) / proj_Mtot
            proj_com += centre
            proj_vcom = (proj_mass[:, None] * proj_velocity).sum(axis=0) / proj_Mtot

            prefix = (
                f"ProjectedAperture/{self.physical_radius_mpc*1000.:.0f}kpc/{projname}"
            )
            halo_result.update(
                {
                    f"{prefix}/Mtot": (proj_Mtot, "Total mass"),
                    f"{prefix}/com": (proj_com, "Centre of mass"),
                    f"{prefix}/vcom": (proj_vcom, "Centre of mass velocity"),
                }
            )

        return
