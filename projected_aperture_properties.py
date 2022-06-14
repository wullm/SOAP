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
            "PartType0": [
                "Coordinates",
                "Velocities",
                "Masses",
                "GroupNr_bound",
                "StarFormationRates",
            ],
            "PartType1": ["Coordinates", "Velocities", "Masses", "GroupNr_bound"],
            "PartType4": [
                "Coordinates",
                "Velocities",
                "Masses",
                "InitialMasses",
                "GroupNr_bound",
                "Luminosities",
            ],
            "PartType5": [
                "Coordinates",
                "Velocities",
                "DynamicalMasses",
                "SubgridMasses",
                "GroupNr_bound",
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
            grnr = data[ptype]["GroupNr_bound"]
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

        for projname, projmask, projr in zip(
            ["projx", "projy", "projz"],
            [mask_projx, mask_projy, mask_projz],
            [radius_projx, radius_projy, radius_projz],
        ):
            proj_mass = mass[projmask]
            proj_position = position[projmask]
            proj_velocity = velocity[projmask]
            proj_radius = projr[projmask]
            proj_type = types[projmask]

            proj_mass_gas = proj_mass[proj_type == "PartType0"]
            proj_mass_dm = proj_mass[proj_type == "PartType1"]
            proj_mass_star = proj_mass[proj_type == "PartType4"]

            proj_Mtot = proj_mass.sum()
            proj_Mgas = proj_mass_gas.sum()
            proj_Mdm = proj_mass_dm.sum()
            proj_Mstar = proj_mass_star.sum()
            if "PartType4" in data:
                star_mask = data["PartType4"]["GroupNr_bound"] == index
                star_mask[star_mask][~projmask[types == "PartType4"]] = False
                proj_Mstar_init = data["PartType4"]["InitialMasses"][star_mask].sum()
                proj_lum = data["PartType4"]["Luminosities"][star_mask].sum(axis=0)
            else:
                proj_Mstar_init = unyt.unyt_array(proj_Mstar)
                proj_lum = unyt.unyt_array(
                    [0.0] * 9,
                    dtype=np.float32,
                    units="dimensionless",
                    registry=mass.units.registry,
                )
            proj_Mbh = proj_mass[proj_type == "PartType5"].sum()
            if "PartType5" in data:
                bh_mask = data["PartType5"]["GroupNr_bound"] == index
                bh_mask[bh_mask][~projmask[types == "PartType5"]] = False
                proj_Mbh_subgrid = data["PartType5"]["SubgridMasses"][bh_mask].sum()
            else:
                proj_Mbh_subgrid = unyt.unyt_array(proj_Mbh)

            proj_com = (proj_mass[:, None] * proj_position).sum(axis=0) / proj_Mtot
            proj_com += centre
            proj_vcom = (proj_mass[:, None] * proj_velocity).sum(axis=0) / proj_Mtot

            if "PartType0" in data:
                gas_mask = data["PartType0"]["GroupNr_bound"] == index
                gas_mask[gas_mask][~projmask[types == "PartType0"]] = False
                proj_SFR = data["PartType0"]["StarFormationRates"][gas_mask].sum()
            else:
                proj_SFR = unyt.unyt_array(
                    0.0, dtype=np.float32, units="Msun/yr", registry=mass.units.registry
                )

            # sort according to radius
            isort_tot = np.argsort(proj_radius)
            isort_gas = np.argsort(proj_radius[proj_type == "PartType0"])
            isort_dm = np.argsort(proj_radius[proj_type == "PartType1"])
            isort_star = np.argsort(proj_radius[proj_type == "PartType4"])
            Mcum_tot = proj_mass[isort_tot].cumsum()
            Mcum_gas = proj_mass_gas[isort_gas].cumsum()
            Mcum_dm = proj_mass_dm[isort_dm].cumsum()
            Mcum_star = proj_mass_star[isort_star].cumsum()
            halfmass = {}
            for name, Mcum, Mtarget, rad in zip(
                ["tot", "gas", "dm", "star"],
                [Mcum_tot, Mcum_gas, Mcum_dm, Mcum_star],
                [0.5 * proj_Mtot, 0.5 * proj_Mgas, 0.5 * proj_Mdm, 0.5 * proj_Mstar],
                [
                    proj_radius[isort_tot],
                    proj_radius[proj_type == "PartType0"][isort_gas],
                    proj_radius[proj_type == "PartType1"][isort_dm],
                    proj_radius[proj_type == "PartType4"][isort_star],
                ],
            ):
                if Mtarget == 0.0 * unyt.Msun or len(Mcum) < 1:
                    halfmass[name] = unyt.unyt_array(
                        0.0, dtype=np.float64, units="kpc", registry=mass.units.registry
                    )
                else:
                    ihalf = np.argmax(Mcum >= Mtarget)
                    if ihalf == 0:
                        # it is possible that we only have one particle, or that there is no
                        # particle with a mass below the target value
                        # in this case, we linearly interpolate from the centre
                        rmin = 0.0 * unyt.kpc
                        Mmin = 0.0 * unyt.Msun
                    else:
                        rmin = rad[ihalf - 1]
                        Mmin = Mcum[ihalf - 1]
                    rmax = rad[ihalf]
                    Mmax = Mcum[ihalf]
                    if Mmin == Mmax:
                        # this deals with the degenerate case where we have no particles below the
                        # target and the first particle above the target is exactly at the centre
                        halfmass[name] = 0.5 * (rmax + rmin)
                    else:
                        halfmass[name] = (Mtarget - Mmin) / (Mmax - Mmin) * (
                            rmax - rmin
                        ) + rmin
                    halfmass[name].convert_to_units("kpc")
                    if halfmass[name] >= self.physical_radius_mpc * unyt.Mpc:
                        raise RuntimeError(
                            "Half mass radius larger than aperture! This should not happen."
                        )

            prefix = (
                f"ProjectedAperture/{self.physical_radius_mpc*1000.:.0f}kpc/{projname}"
            )
            halo_result.update(
                {
                    f"{prefix}/Mtot": (proj_Mtot, "Total mass"),
                    f"{prefix}/Mgas": (proj_Mgas, "Total gas mass"),
                    f"{prefix}/Mdm": (proj_Mdm, "Total DM mass"),
                    f"{prefix}/Mstar": (proj_Mstar, "Total stellar mass"),
                    f"{prefix}/Mstar_init": (
                        proj_Mstar_init,
                        "Total initial stellar mass",
                    ),
                    f"{prefix}/Mbh": (proj_Mbh, "Total BH dynamical mass"),
                    f"{prefix}/Mbh_subgrid": (
                        proj_Mbh_subgrid,
                        "Total BH subgrid mass",
                    ),
                    f"{prefix}/com": (proj_com, "Centre of mass"),
                    f"{prefix}/vcom": (proj_vcom, "Centre of mass velocity"),
                    f"{prefix}/SFR": (proj_SFR, "Total SFR"),
                    f"{prefix}/Luminosity": (proj_lum, "Total luminosity"),
                    f"{prefix}/HalfMassRadiusTot": (
                        halfmass["tot"],
                        "Total half mass radius",
                    ),
                    f"{prefix}/HalfMassRadiusGas": (
                        halfmass["gas"],
                        "Total gas half mass radius",
                    ),
                    f"{prefix}/HalfMassRadiusDM": (
                        halfmass["dm"],
                        "Total DM half mass radius",
                    ),
                    f"{prefix}/HalfMassRadiusStar": (
                        halfmass["star"],
                        "Total stellar half mass radius",
                    ),
                }
            )

        return
