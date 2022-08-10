##!/bin/env python

import numpy as np
import unyt

from halo_properties import HaloProperty
from dataset_names import mass_dataset
from half_mass_radius import get_half_mass_radius
from property_table import PropertyTable


class ProjectedApertureProperties(HaloProperty):
    """
    Calculate projected aperture properties.

    These contain all particles bound to a halo. For projections along the three
    principal coordinate axes, all particles within a given fixed aperture
    radius are used. The depth of the projection is always the full extent of
    the halo along the projection axis.
    """

    # get the properties we want from the table
    property_list = [
        (prop, *PropertyTable.full_property_list[prop])
        for prop in [
            "Mtot",
            "Mgas",
            "Mdm",
            "Mstar",
            "Mstar_init",
            "Mbh_dynamical",
            "Mbh_subgrid",
            "Ngas",
            "Ndm",
            "Nstar",
            "Nbh",
            "com",
            "vcom",
            "SFR",
            "StellarLuminosity",
            "HalfMassRadiusGas",
            "HalfMassRadiusDM",
            "HalfMassRadiusStar",
            "proj_veldisp_gas",
            "proj_veldisp_dm",
            "proj_veldisp_star",
            "BHmaxM",
            "BHmaxID",
            "BHmaxpos",
            "BHmaxvel",
            "BHlasteventa",
            "BHmaxlasteventa",
        ]
    ]

    # Particle properties that are used
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
            "LastAGNFeedbackScaleFactors",
            "ParticleIDs",
            "SubgridMasses",
            "Velocities",
        ],
    }

    def __init__(self, cellgrid, physical_radius_kpc):
        super().__init__(cellgrid)

        # No density criterion
        self.mean_density_multiple = None
        self.critical_density_multiple = None

        # Minimum physical radius to read in (pMpc)
        self.physical_radius_mpc = 0.001 * physical_radius_kpc

        self.name = f"projected_aperture_{physical_radius_kpc:.0f}kpc"

    def calculate(self, input_halo, search_radius, data, halo_result):
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

        for iproj, (projname, projmask, projr) in enumerate(
            zip(
                ["projx", "projy", "projz"],
                [mask_projx, mask_projy, mask_projz],
                [radius_projx, radius_projy, radius_projz],
            )
        ):

            projected_aperture = {}
            # declare all the variables we will compute
            # we set them to 0 in case a particular variable cannot be computed
            # all variables are defined with physical units and an appropriate dtype
            # we need to use the custom unit registry so that everything can be converted
            # back to snapshot units in the end
            for name, _, shape, dtype, unit, _, _ in self.property_list:
                if shape > 1:
                    val = [0] * shape
                else:
                    val = 0
                projected_aperture[name] = unyt.unyt_array(
                    val, dtype=dtype, units=unit, registry=mass.units.registry
                )

            proj_mass = mass[projmask]
            proj_position = position[projmask]
            proj_velocity = velocity[projmask]
            proj_radius = projr[projmask]
            proj_type = types[projmask]

            gas_mask_ap = projmask[types == "PartType0"]
            star_mask_ap = projmask[types == "PartType4"]
            bh_mask_ap = projmask[types == "PartType5"]

            projected_aperture["Ngas"] = (
                gas_mask_ap.sum(dtype=projected_aperture["Ngas"].dtype)
                * projected_aperture["Ngas"].units
            )
            projected_aperture["Ndm"] = (
                projmask[types == "PartType1"].sum(
                    dtype=projected_aperture["Ndm"].dtype
                )
                * projected_aperture["Ndm"].units
            )
            projected_aperture["Nstar"] = (
                star_mask_ap.sum(dtype=projected_aperture["Nstar"].dtype)
                * projected_aperture["Nstar"].units
            )
            projected_aperture["Nbh"] = (
                bh_mask_ap.sum(dtype=projected_aperture["Nbh"].dtype)
                * projected_aperture["Nbh"].units
            )

            proj_mass_gas = proj_mass[proj_type == "PartType0"]
            proj_mass_dm = proj_mass[proj_type == "PartType1"]
            proj_mass_star = proj_mass[proj_type == "PartType4"]

            projected_aperture["Mtot"] += proj_mass.sum()
            projected_aperture["Mgas"] += proj_mass_gas.sum()
            projected_aperture["Mdm"] += proj_mass_dm.sum()
            projected_aperture["Mstar"] += proj_mass_star.sum()
            if np.any(star_mask_ap):
                star_mask_all = data["PartType4"]["GroupNr_bound"] == index
                projected_aperture["Mstar_init"] += data["PartType4"]["InitialMasses"][
                    star_mask_all
                ][star_mask_ap].sum()
                projected_aperture["StellarLuminosity"] += data["PartType4"][
                    "Luminosities"
                ][star_mask_all][star_mask_ap].sum(axis=0)

            projected_aperture["Mbh_dynamical"] += proj_mass[
                proj_type == "PartType5"
            ].sum()
            if np.any(bh_mask_ap):
                bh_mask_all = data["PartType5"]["GroupNr_bound"] == index
                bh_masses = data["PartType5"]["SubgridMasses"][bh_mask_all][bh_mask_ap]
                projected_aperture["Mbh_subgrid"] += bh_masses.sum()
                iBHmax = np.argmax(bh_masses)
                projected_aperture["BHmaxM"] += bh_masses[iBHmax]
                projected_aperture["BHmaxID"] = (
                    data["PartType5"]["ParticleIDs"][bh_mask_all][bh_mask_ap][
                        iBHmax
                    ].astype(projected_aperture["BHmaxID"].dtype)
                    * projected_aperture["BHmaxID"].units
                )
                agn_eventa = data["PartType5"]["LastAGNFeedbackScaleFactors"][
                    bh_mask_all
                ][bh_mask_ap]
                projected_aperture["BHlasteventa"] += np.max(agn_eventa)
                projected_aperture["BHmaxlasteventa"] += agn_eventa[iBHmax]
                projected_aperture["BHmaxpos"] += data["PartType5"]["Coordinates"][
                    bh_mask_all
                ][bh_mask_ap][iBHmax]
                projected_aperture["BHmaxvel"] += data["PartType5"]["Velocities"][
                    bh_mask_all
                ][bh_mask_ap][iBHmax]

            if projected_aperture["Mtot"] > 0.0 * projected_aperture["Mtot"].units:
                mass_frac = proj_mass / projected_aperture["Mtot"]
                projected_aperture["com"] += (mass_frac[:, None] * proj_position).sum(
                    axis=0
                )
                projected_aperture["com"] += centre
                # perform the mass division before the multiplication to avoid
                # numerical overflow
                projected_aperture["vcom"] += (mass_frac[:, None] * proj_velocity).sum(
                    axis=0
                )

            if np.any(gas_mask_ap):
                gas_mask_all = data["PartType0"]["GroupNr_bound"] == index
                proj_SFR = data["PartType0"]["StarFormationRates"][gas_mask_all][
                    gas_mask_ap
                ]
                # Negative SFR are not SFR at all!
                projected_aperture["SFR"] += proj_SFR[proj_SFR > 0.0].sum()

            if projected_aperture["Mgas"] > 0.0 * projected_aperture["Mgas"].units:
                frac_mgas = proj_mass_gas / projected_aperture["Mgas"]
                proj_vgas = proj_velocity[proj_type == "PartType0", iproj]
                vcom_gas = (frac_mgas * proj_vgas).sum()
                projected_aperture["proj_veldisp_gas"] += np.sqrt(
                    ((proj_vgas - vcom_gas) ** 2).sum()
                )
            if projected_aperture["Mdm"] > 0.0 * projected_aperture["Mdm"].units:
                frac_mdm = proj_mass_dm / projected_aperture["Mdm"]
                proj_vdm = proj_velocity[proj_type == "PartType1", iproj]
                vcom_dm = (frac_mdm * proj_vdm).sum()
                projected_aperture["proj_veldisp_dm"] += np.sqrt(
                    ((proj_vdm - vcom_dm) ** 2).sum()
                )
            if projected_aperture["Mstar"] > 0.0 * projected_aperture["Mstar"].units:
                frac_mstar = proj_mass_star / projected_aperture["Mstar"]
                proj_vstar = proj_velocity[proj_type == "PartType4", iproj]
                vcom_star = (frac_mstar * proj_vstar).sum()
                projected_aperture["proj_veldisp_star"] += np.sqrt(
                    ((proj_vstar - vcom_star) ** 2).sum()
                )

            for name, r, m, M in zip(
                [
                    "HalfMassRadiusGas",
                    "HalfMassRadiusDM",
                    "HalfMassRadiusStar",
                ],
                [
                    proj_radius[proj_type == "PartType0"],
                    proj_radius[proj_type == "PartType1"],
                    proj_radius[proj_type == "PartType4"],
                ],
                [proj_mass_gas, proj_mass_dm, proj_mass_star],
                [
                    projected_aperture["Mgas"],
                    projected_aperture["Mdm"],
                    projected_aperture["Mstar"],
                ],
            ):
                projected_aperture[name] += get_half_mass_radius(r, m, M)
                if projected_aperture[name] >= self.physical_radius_mpc * unyt.Mpc:
                    raise RuntimeError(
                        "Half mass radius larger than aperture"
                        f" ({half_mass_radius} >="
                        f" {self.physical_radius_mpc * unyt.Mpc}!"
                        " This should not happen."
                    )

            prefix = (
                f"ProjectedAperture/{self.physical_radius_mpc*1000.:.0f}kpc/{projname}"
            )
            for name, outputname, _, _, _, description, _ in self.property_list:
                halo_result.update(
                    {f"{prefix}/{outputname}": (projected_aperture[name], description)}
                )

        return


class DummyProjectedApertureProperties(ProjectedApertureProperties):
    """
    Minimal version of ProjectedApertureProperties, only used for testing.
    """

    def __init__(self):
        self.physical_radius_mpc = 0.03


def test_projected_aperture_properties():
    """
    Unit test for the projected aperture calculation.

    Generates 100 random halos and passes them on to
    ProjectedApertureProperties::calculate().
    Tests that all expected return values are computed and have the right size,
    dtype and units.
    """

    from dummy_halo_generator import DummyHaloGenerator

    dummy_halos = DummyHaloGenerator(127)

    property_calculator = DummyProjectedApertureProperties()

    for i in range(100):
        input_halo, data, _, _, _ = dummy_halos.get_random_halo(
            [1, 10, 100, 1000, 10000]
        )

        input_data = {}
        for ptype in property_calculator.particle_properties:
            if ptype in data:
                input_data[ptype] = {}
                for dset in property_calculator.particle_properties[ptype]:
                    input_data[ptype][dset] = data[ptype][dset]
        input_halo_copy = input_halo.copy()
        input_data_copy = input_data.copy()
        halo_result = {}
        property_calculator.calculate(
            input_halo, 0.0 * unyt.kpc, input_data, halo_result
        )
        assert input_halo == input_halo_copy
        assert input_data == input_data_copy

        for proj in ["projx", "projy", "projz"]:
            for (
                _,
                outputname,
                size,
                dtype,
                unit_string,
                _,
                _,
            ) in property_calculator.property_list:
                full_name = f"ProjectedAperture/30kpc/{proj}/{outputname}"
                assert full_name in halo_result
                result = halo_result[full_name][0]
                assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
                assert result.dtype == dtype
                unit = unyt.Unit(unit_string)
                assert result.units.same_dimensions_as(unit.units)


if __name__ == "__main__":
    """
    Standalone mode: simply run the unit test.

    Note that this can also be achieved by running
    python3 -m pytest *.py
    in the main folder.
    """
    print("Calling test_projected_aperture_properties()...")
    test_projected_aperture_properties()
    print("Test passed.")
