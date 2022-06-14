#!/bin/env python

import numpy as np
import unyt

from halo_properties import HaloProperty
from dataset_names import mass_dataset


class ExclusiveSphereProperties(HaloProperty):
    def __init__(self, cellgrid, physical_radius_kpc):
        super().__init__(cellgrid)

        # This specifies how large a sphere is read in:
        self.mean_density_multiple = None
        self.critical_density_multiple = None

        # Minimum physical radius to read in (pMpc)
        self.physical_radius_mpc = 0.001 * physical_radius_kpc

        self.name = f"exclusive_sphere_{physical_radius_kpc:.0f}kpc"

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
        radius = []
        velocity = []
        types = []
        for ptype in types_present:
            grnr = data[ptype]["GroupNr_bound"]
            in_halo = grnr == index
            mass.append(data[ptype][mass_dataset(ptype)][in_halo])
            pos = data[ptype]["Coordinates"][in_halo, :] - centre[None, :]
            position.append(pos)
            r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2)
            radius.append(r)
            velocity.append(data[ptype]["Velocities"][in_halo, :])
            typearr = np.zeros(r.shape, dtype="U9")
            typearr[:] = ptype
            types.append(typearr)

        mass = unyt.array.uconcatenate(mass)
        position = unyt.array.uconcatenate(position)
        radius = unyt.array.uconcatenate(radius)
        velocity = unyt.array.uconcatenate(velocity)
        types = np.concatenate(types)

        mask = radius <= self.physical_radius_mpc * unyt.Mpc

        mass = mass[mask]
        position = position[mask]
        velocity = velocity[mask]
        radius = radius[mask]
        type = types[mask]

        mass_gas = mass[type == "PartType0"]
        mass_dm = mass[type == "PartType1"]
        mass_star = mass[type == "PartType4"]

        Mtot = mass.sum()
        Mgas = mass_gas.sum()
        Mdm = mass_dm.sum()
        Mstar = mass_star.sum()
        if "PartType4" in data:
            star_mask = data["PartType4"]["GroupNr_bound"] == index
            star_mask[star_mask][~mask[types == "PartType4"]] = False
            Mstar_init = data["PartType4"]["InitialMasses"][star_mask].sum()
            lum = data["PartType4"]["Luminosities"][star_mask].sum(axis=0)
        else:
            Mstar_init = unyt.unyt_array(Mstar, dtype=Mstar.dtype, units=Mstar.units)
            lum = unyt.unyt_array(
                [0.0] * 9,
                dtype=np.float32,
                units="dimensionless",
                registry=mass.units.registry,
            )
        Mbh = mass[type == "PartType5"].sum()
        if "PartType5" in data:
            bh_mask = data["PartType5"]["GroupNr_bound"] == index
            bh_mask[bh_mask][~mask[types == "PartType5"]] = False
            Mbh_subgrid = data["PartType5"]["SubgridMasses"][bh_mask].sum()
        else:
            Mbh_subgrid = unyt.unyt_array(Mbh, dtype=Mbh.dtype, units=Mbh.units)

        com = (mass[:, None] * position).sum(axis=0) / Mtot
        com += centre
        vcom = (mass[:, None] * velocity).sum(axis=0) / Mtot
        if "PartType0" in data:
            gas_mask = data["PartType0"]["GroupNr_bound"] == index
            gas_mask[gas_mask][~mask[types == "PartType0"]] = False
            SFR = data["PartType0"]["StarFormationRates"][gas_mask].sum()
        else:
            SFR = unyt.unyt_array(
                0.0, dtype=np.float32, units="Msun/yr", registry=mass.units.registry
            )

        # sort according to radius
        isort_tot = np.argsort(radius)
        isort_gas = np.argsort(radius[type == "PartType0"])
        isort_dm = np.argsort(radius[type == "PartType1"])
        isort_star = np.argsort(radius[type == "PartType4"])
        Mcum_tot = mass[isort_tot].cumsum()
        Mcum_gas = mass_gas[isort_gas].cumsum()
        Mcum_dm = mass_dm[isort_dm].cumsum()
        Mcum_star = mass_star[isort_star].cumsum()
        halfmass = {}
        for name, Mcum, Mtarget, rad in zip(
            ["tot", "gas", "dm", "star"],
            [Mcum_tot, Mcum_gas, Mcum_dm, Mcum_star],
            [0.5 * Mtot, 0.5 * Mgas, 0.5 * Mdm, 0.5 * Mstar],
            [
                radius[isort_tot],
                radius[type == "PartType0"][isort_gas],
                radius[type == "PartType1"][isort_dm],
                radius[type == "PartType4"][isort_star],
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
                    # note that the unit registry of the first variable in the
                    # expression is used for the result. If rmin was set to
                    # 0, we do not want to use its registry. We rearranged the
                    # expression to enforce the correct registry.
                    halfmass[name] = (Mtarget - Mmin) / (Mmax - Mmin) * (
                        rmax - rmin
                    ) + rmin
                halfmass[name].convert_to_units("kpc")
                if halfmass[name] >= self.physical_radius_mpc * unyt.Mpc:
                    raise RuntimeError(
                        "Half mass radius larger than aperture! This should not happen."
                    )

        prefix = f"ExclusiveSphere/{self.physical_radius_mpc*1000.:.0f}kpc"
        halo_result.update(
            {
                f"{prefix}/Mtot": (Mtot, "Total mass"),
                f"{prefix}/Mgas": (Mgas, "Total gas mass"),
                f"{prefix}/Mdm": (Mdm, "Total DM mass"),
                f"{prefix}/Mstar": (Mstar, "Total stellar mass"),
                f"{prefix}/Mstar_init": (
                    Mstar_init,
                    "Total initial stellar mass",
                ),
                f"{prefix}/Mbh": (Mbh, "Total BH dynamical mass"),
                f"{prefix}/Mbh_subgrid": (
                    Mbh_subgrid,
                    "Total BH subgrid mass",
                ),
                f"{prefix}/com": (com, "Centre of mass"),
                f"{prefix}/vcom": (vcom, "Centre of mass velocity"),
                f"{prefix}/SFR": (SFR, "Total SFR"),
                f"{prefix}/Luminosity": (lum, "Total luminosity"),
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
