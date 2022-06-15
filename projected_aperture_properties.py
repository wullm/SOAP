#!/bin/env python

import numpy as np
import unyt

from halo_properties import HaloProperty
from dataset_names import mass_dataset
from half_mass_radius import get_half_mass_radius


class ProjectedApertureProperties(HaloProperty):
    particle_properties = {
        "PartType0": [
            "Coordinates",
            "GroupNr_bound",
            "Masses",
            "StarFormationRates",
            "Velocities",
        ],
        "PartType1": ["Coordinates", "GroupNr_bound", "Masses", "Velocities"],
        "PartType4": [
            "Coordinates",
            "GroupNr_bound",
            "InitialMasses",
            "Luminosities",
            "Masses",
            "Velocities",
        ],
        "PartType5": [
            "Coordinates",
            "DynamicalMasses",
            "GroupNr_bound",
            "SubgridMasses",
            "Velocities",
        ],
    }

    def __init__(self, cellgrid, physical_radius_kpc):
        super().__init__(cellgrid)

        # This specifies how large a sphere is read in:
        self.mean_density_multiple = None
        self.critical_density_multiple = None

        # Minimum physical radius to read in (pMpc)
        self.physical_radius_mpc = 0.001 * physical_radius_kpc

        self.name = f"projected_aperture_{physical_radius_kpc:.0f}kpc"

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

            gas_mask_ap = projmask[types == "PartType0"]
            star_mask_ap = projmask[types == "PartType4"]
            bh_mask_ap = projmask[types == "PartType5"]

            proj_mass_gas = proj_mass[proj_type == "PartType0"]
            proj_mass_dm = proj_mass[proj_type == "PartType1"]
            proj_mass_star = proj_mass[proj_type == "PartType4"]

            proj_Mtot = proj_mass.sum()
            proj_Mgas = proj_mass_gas.sum()
            proj_Mdm = proj_mass_dm.sum()
            proj_Mstar = proj_mass_star.sum()
            if np.any(star_mask_ap):
                star_mask_all = data["PartType4"]["GroupNr_bound"] == index
                proj_Mstar_init = data["PartType4"]["InitialMasses"][star_mask_all][
                    star_mask_ap
                ].sum()
                proj_lum = data["PartType4"]["Luminosities"][star_mask_all][
                    star_mask_ap
                ].sum(axis=0)
            else:
                proj_Mstar_init = unyt.unyt_array(proj_Mstar)
                proj_lum = unyt.unyt_array(
                    [0.0] * 9,
                    dtype=np.float32,
                    units="dimensionless",
                    registry=mass.units.registry,
                )
            proj_Mbh = proj_mass[proj_type == "PartType5"].sum()
            if np.any(bh_mask_ap):
                bh_mask_all = data["PartType5"]["GroupNr_bound"] == index
                proj_Mbh_subgrid = data["PartType5"]["SubgridMasses"][bh_mask_all][
                    bh_mask_ap
                ].sum()
            else:
                proj_Mbh_subgrid = unyt.unyt_array(proj_Mbh)

            proj_com = unyt.unyt_array(
                [0.0] * 3, dtype=np.float32, units="Mpc", registry=mass.units.registry
            )
            proj_vcom = unyt.unyt_array(
                [0.0] * 3, dtype=np.float32, units="km/s", registry=mass.units.registry
            )
            if proj_Mtot > 0.0 * proj_Mtot.units:
                proj_com[:] = (proj_mass[:, None] * proj_position).sum(
                    axis=0
                ) / proj_Mtot
                proj_com[:] += centre
                # perform the mass division before the multiplication to avoid
                # numerical overflow
                proj_vcom[:] = ((proj_mass[:, None] / proj_Mtot) * proj_velocity).sum(
                    axis=0
                )

            proj_SFR = 0.0
            if np.any(gas_mask_ap):
                gas_mask_all = data["PartType0"]["GroupNr_bound"] == index
                proj_SFR = data["PartType0"]["StarFormationRates"][gas_mask_all][
                    gas_mask_ap
                ]
                # Negative SFR are not SFR at all!
                proj_SFR = proj_SFR[proj_SFR > 0.0].sum()
            proj_SFR = unyt.unyt_array(
                proj_SFR,
                dtype=np.float32,
                units="Msun/yr",
                registry=mass.units.registry,
            )

            # sort according to radius
            halfmass = {}
            for name, r, m, M in zip(
                ["tot", "gas", "dm", "star"],
                [
                    proj_radius,
                    proj_radius[proj_type == "PartType0"],
                    proj_radius[proj_type == "PartType1"],
                    proj_radius[proj_type == "PartType4"],
                ],
                [proj_mass, proj_mass_gas, proj_mass_dm, proj_mass_star],
                [proj_Mtot, proj_Mgas, proj_Mdm, proj_Mstar],
            ):
                half_mass_radius = get_half_mass_radius(r, m, M)
                if half_mass_radius >= self.physical_radius_mpc * unyt.Mpc:
                    raise RuntimeError(
                        "Half mass radius larger than aperture"
                        f" ({half_mass_radius} >="
                        f" {self.physical_radius_mpc * unyt.Mpc}!"
                        " This should not happen."
                    )
                halfmass[name] = unyt.unyt_array(
                    half_mass_radius.value,
                    dtype=np.float32,
                    units=half_mass_radius.units,
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


class DummyProjectedApertureProperties(ProjectedApertureProperties):
    def __init__(self):
        self.physical_radius_mpc = 0.001


def test_projected_aperture_properties():

    property_calculator = DummyProjectedApertureProperties()

    np.random.seed(127)
    for i in range(100):
        npart = np.random.choice([1, 10, 100, 1000, 10000])

        centre = 100.0 * np.random.random(3) * unyt.Mpc
        groupnr_halo = 1

        radius = np.random.exponential(1.0, npart)
        phi = 2.0 * np.pi * np.random.random(npart)
        sintheta = 2.0 * np.random.random(npart) - 1.0
        costheta = np.sqrt((1.0 - sintheta) * (1.0 + sintheta))
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        coords = np.zeros((npart, 3))
        coords[:, 0] = radius * cosphi * sintheta
        coords[:, 1] = radius * sinphi * sintheta
        coords[:, 2] = radius * costheta
        coords = unyt.unyt_array(coords, dtype=np.float64, units=unyt.kpc)
        coords += centre
        mass = unyt.unyt_array(
            1.0e9 * (1.0 - 0.2 * np.random.random(npart)),
            dtype=np.float32,
            units=unyt.Msun,
        )
        vs = unyt.unyt_array(
            200.0 * (np.random.random((npart, 3)) - 0.5),
            dtype=np.float32,
            units=unyt.km / unyt.s,
        )

        types = np.random.choice(
            ["PartType0", "PartType1", "PartType4", "PartType5"], size=npart
        )
        groupnr = np.random.choice([groupnr_halo, 2, 3], size=npart)

        data = {}
        gas_mask = types == "PartType0"
        Ngas = int(gas_mask.sum())
        if Ngas > 0:
            data["PartType0"] = {}
            data["PartType0"]["Coordinates"] = coords[gas_mask]
            data["PartType0"]["GroupNr_bound"] = groupnr[gas_mask]
            data["PartType0"]["Masses"] = mass[gas_mask]
            data["PartType0"]["StarFormationRates"] = unyt.unyt_array(
                (np.random.random(Ngas) - 0.5),
                dtype=np.float32,
                units=unyt.Msun / unyt.yr,
            )
            data["PartType0"]["Velocities"] = vs[gas_mask]

        dm_mask = types == "PartType1"
        Ndm = int(dm_mask.sum())
        if Ndm > 0:
            data["PartType1"] = {}
            data["PartType1"]["Coordinates"] = coords[dm_mask]
            data["PartType1"]["GroupNr_bound"] = groupnr[dm_mask]
            data["PartType1"]["Masses"] = mass[dm_mask]
            data["PartType1"]["Velocities"] = vs[dm_mask]

        star_mask = types == "PartType4"
        Nstar = int(star_mask.sum())
        if Nstar > 0:
            data["PartType4"] = {}
            data["PartType4"]["Coordinates"] = coords[star_mask]
            data["PartType4"]["GroupNr_bound"] = groupnr[star_mask]
            data["PartType4"]["InitialMasses"] = unyt.unyt_array(
                mass[star_mask].value * (0.9 + 0.1 * np.random.random(Nstar)),
                dtype=np.float32,
                units=unyt.Msun,
            )
            data["PartType4"]["Luminosities"] = unyt.unyt_array(
                np.random.random((Nstar, 9)), dtype=np.float32, units=unyt.dimensionless
            )
            data["PartType4"]["Masses"] = mass[star_mask]
            data["PartType4"]["Velocities"] = vs[star_mask]

        bh_mask = types == "PartType5"
        Nbh = int(bh_mask.sum())
        if Nbh > 0:
            data["PartType5"] = {}
            data["PartType5"]["Coordinates"] = coords[bh_mask]
            data["PartType5"]["DynamicalMasses"] = mass[bh_mask]
            data["PartType5"]["GroupNr_bound"] = groupnr[bh_mask]
            data["PartType5"]["SubgridMasses"] = unyt.unyt_array(
                mass[bh_mask].value * (0.9 + 0.1 * np.random.random(Nbh)),
                dtype=np.float32,
                units=unyt.Msun,
            )
            data["PartType5"]["Velocities"] = vs[bh_mask]

        input_halo = {}
        input_halo["cofp"] = centre
        input_halo["index"] = groupnr_halo

        halo_result = {}

        property_calculator.calculate(input_halo, data, halo_result)

        for name, size, dtype, unit in [
            ("ProjectedAperture/1kpc/projx/Mtot", 1, np.float32, unyt.Msun),
            ("ProjectedAperture/1kpc/projx/Mgas", 1, np.float32, unyt.Msun),
            ("ProjectedAperture/1kpc/projx/Mdm", 1, np.float32, unyt.Msun),
            ("ProjectedAperture/1kpc/projx/Mstar", 1, np.float32, unyt.Msun),
            ("ProjectedAperture/1kpc/projx/Mstar_init", 1, np.float32, unyt.Msun),
            ("ProjectedAperture/1kpc/projx/Mbh", 1, np.float32, unyt.Msun),
            ("ProjectedAperture/1kpc/projx/Mbh_subgrid", 1, np.float32, unyt.Msun),
            ("ProjectedAperture/1kpc/projx/com", 3, np.float32, unyt.kpc),
            ("ProjectedAperture/1kpc/projx/vcom", 3, np.float32, unyt.km / unyt.s),
            ("ProjectedAperture/1kpc/projx/SFR", 1, np.float32, unyt.Msun / unyt.yr),
            (
                "ProjectedAperture/1kpc/projx/Luminosity",
                9,
                np.float32,
                unyt.dimensionless,
            ),
            ("ProjectedAperture/1kpc/projx/HalfMassRadiusTot", 1, np.float32, unyt.kpc),
            ("ProjectedAperture/1kpc/projx/HalfMassRadiusGas", 1, np.float32, unyt.kpc),
            ("ProjectedAperture/1kpc/projx/HalfMassRadiusDM", 1, np.float32, unyt.kpc),
            (
                "ProjectedAperture/1kpc/projx/HalfMassRadiusStar",
                1,
                np.float32,
                unyt.kpc,
            ),
        ]:
            assert name in halo_result
            result = halo_result[name][0]
            assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
            assert result.dtype == dtype
            assert result.units.same_dimensions_as(unit.units)
