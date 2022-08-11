#!/bin/env python

import numpy as np
import unyt

from halo_properties import HaloProperty
from dataset_names import mass_dataset
from half_mass_radius import get_half_mass_radius
from kinematic_properties import (
    get_velocity_dispersion_matrix,
    get_angular_momentum,
    get_angular_momentum_and_kappa_corot,
    get_vmax,
    get_axis_lengths,
)
from recently_heated_gas_filter import RecentlyHeatedGasFilter
from property_table import PropertyTable

# index of elements O and Fe in the SmoothedElementMassFractions dataset
indexO = 4
indexFe = 8


class ExclusiveSphereProperties(HaloProperty):
    """
    Compute exclusive sphere properties for halos.

    The exclusive sphere has a fixed radius and only includes particles that
    are bound to the halo.
    """

    # List of particle properties we need to read in
    particle_properties = {
        "PartType0": [
            "Coordinates",
            "GroupNr_bound",
            "LastAGNFeedbackScaleFactors",
            "Masses",
            "MetalMassFractions",
            "SmoothedElementMassFractions",
            "StarFormationRates",
            "Temperatures",
            "Velocities",
        ],
        "PartType1": ["Coordinates", "GroupNr_bound", "Masses", "Velocities"],
        "PartType4": [
            "Coordinates",
            "GroupNr_bound",
            "InitialMasses",
            "Luminosities",
            "Masses",
            "MetalMassFractions",
            "SmoothedElementMassFractions",
            "Velocities",
        ],
        "PartType5": [
            "AccretionRates",
            "Coordinates",
            "DynamicalMasses",
            "GroupNr_bound",
            "LastAGNFeedbackScaleFactors",
            "ParticleIDs",
            "SubgridMasses",
            "Velocities",
        ],
    }

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
            "BHlasteventa",
            "BHmaxM",
            "BHmaxID",
            "BHmaxpos",
            "BHmaxvel",
            "BHmaxAR",
            "BHmaxlasteventa",
            "com",
            "vcom",
            "Lgas",
            "Ldm",
            "Lstar",
            "kappa_corot_gas",
            "kappa_corot_star",
            "Lbaryons",
            "kappa_corot_baryons",
            # temporarily (?) disabled
            #            "veldisp_matrix_gas",
            #            "veldisp_matrix_dm",
            #            "veldisp_matrix_star",
            "Ekin_gas",
            "Ekin_star",
            "Mgas_SF",
            "Mgasmetal",
            "Mgasmetal_SF",
            "MgasO",
            "MgasO_SF",
            "MgasFe",
            "MgasFe_SF",
            "Tgas",
            "Tgas_no_agn",
            "SFR",
            "StellarLuminosity",
            "Mstarmetal",
            "HalfMassRadiusGas",
            "HalfMassRadiusDM",
            "HalfMassRadiusStar",
            "spin_parameter",
            "GasAxisLengths",
            "DMAxisLengths",
            "StellarAxisLengths",
            "BaryonAxisLengths",
            "DtoTgas",
            "DtoTstar",
            "MstarO",
            "MstarFe",
        ]
    ]

    def __init__(self, cellgrid, physical_radius_kpc, recently_heated_gas_filter):
        """
        Construct an ExclusiveSphereProperties object with the given physical
        radius (in Mpc) that uses the given filter to filter out recently
        heated gas particles.
        """

        super().__init__(cellgrid)

        self.filter = recently_heated_gas_filter

        # no density criterion for these properties
        self.mean_density_multiple = None
        self.critical_density_multiple = None

        # Minimum physical radius to read in (pMpc)
        self.physical_radius_mpc = 0.001 * physical_radius_kpc

        self.name = f"exclusive_sphere_{physical_radius_kpc:.0f}kpc"

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

        exclusive_sphere = {}
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
            exclusive_sphere[name] = unyt.unyt_array(
                val, dtype=dtype, units=unit, registry=mass.units.registry
            )

        mask = radius <= self.physical_radius_mpc * unyt.Mpc

        mass = mass[mask]
        position = position[mask]
        velocity = velocity[mask]
        radius = radius[mask]
        type = types[mask]

        gas_mask_ap = mask[types == "PartType0"]
        dm_mask_ap = mask[types == "PartType1"]
        star_mask_ap = mask[types == "PartType4"]
        bh_mask_ap = mask[types == "PartType5"]
        baryons_mask_ap = mask[(types == "PartType0") | (types == "PartType4")]

        exclusive_sphere["Ngas"] = (
            gas_mask_ap.sum(dtype=exclusive_sphere["Ngas"].dtype)
            * exclusive_sphere["Ngas"].units
        )
        exclusive_sphere["Ndm"] = (
            dm_mask_ap.sum(dtype=exclusive_sphere["Ndm"].dtype)
            * exclusive_sphere["Ndm"].units
        )
        exclusive_sphere["Nstar"] = (
            star_mask_ap.sum(dtype=exclusive_sphere["Nstar"].dtype)
            * exclusive_sphere["Nstar"].units
        )
        exclusive_sphere["Nbh"] = (
            bh_mask_ap.sum(dtype=exclusive_sphere["Nbh"].dtype)
            * exclusive_sphere["Nbh"].units
        )

        mass_gas = mass[type == "PartType0"]
        mass_dm = mass[type == "PartType1"]
        mass_star = mass[type == "PartType4"]
        mass_baryons = mass[(type == "PartType0") | (type == "PartType4")]

        pos_gas = position[type == "PartType0"]
        pos_dm = position[type == "PartType1"]
        pos_star = position[type == "PartType4"]
        pos_baryons = position[(type == "PartType0") | (type == "PartType4")]

        vel_gas = velocity[type == "PartType0"]
        vel_dm = velocity[type == "PartType1"]
        vel_star = velocity[type == "PartType4"]
        vel_baryons = velocity[(type == "PartType0") | (type == "PartType4")]

        exclusive_sphere["Mtot"] += mass.sum()
        exclusive_sphere["Mgas"] += mass_gas.sum()
        exclusive_sphere["Mdm"] = mass_dm.sum()
        exclusive_sphere["Mstar"] += mass_star.sum()
        if exclusive_sphere["Nstar"] > 0:
            star_mask_all = data["PartType4"]["GroupNr_bound"] == index
            exclusive_sphere["Mstar_init"] += data["PartType4"]["InitialMasses"][
                star_mask_all
            ][star_mask_ap].sum()
            exclusive_sphere["StellarLuminosity"] += data["PartType4"]["Luminosities"][
                star_mask_all
            ][star_mask_ap].sum(axis=0)
            exclusive_sphere["Mstarmetal"] += (
                mass_star
                * data["PartType4"]["MetalMassFractions"][star_mask_all][star_mask_ap]
            ).sum()
            MstarO = (
                mass_star
                * data["PartType4"]["SmoothedElementMassFractions"][star_mask_all][
                    star_mask_ap
                ][:, indexO]
            )
            exclusive_sphere["MstarO"] += MstarO.sum()
            MstarFe = (
                mass_star
                * data["PartType4"]["SmoothedElementMassFractions"][star_mask_all][
                    star_mask_ap
                ][:, indexFe]
            )
            exclusive_sphere["MstarFe"] += MstarFe.sum()
        exclusive_sphere["Mbh_dynamical"] = mass[type == "PartType5"].sum()
        if exclusive_sphere["Nbh"] > 0:
            bh_mask_all = data["PartType5"]["GroupNr_bound"] == index
            exclusive_sphere["Mbh_subgrid"] += data["PartType5"]["SubgridMasses"][
                bh_mask_all
            ][bh_mask_ap].sum()

            agn_eventa = data["PartType5"]["LastAGNFeedbackScaleFactors"][bh_mask_all][
                bh_mask_ap
            ]

            exclusive_sphere["BHlasteventa"] += np.max(agn_eventa)

            iBHmax = np.argmax(
                data["PartType5"]["SubgridMasses"][bh_mask_all][bh_mask_ap]
            )
            exclusive_sphere["BHmaxM"] += data["PartType5"]["SubgridMasses"][
                bh_mask_all
            ][bh_mask_ap][iBHmax]
            exclusive_sphere["BHmaxID"] = (
                data["PartType5"]["ParticleIDs"][bh_mask_all][bh_mask_ap][
                    iBHmax
                ].astype(exclusive_sphere["BHmaxID"].dtype)
                * exclusive_sphere["BHmaxID"].units
            )
            exclusive_sphere["BHmaxpos"] += data["PartType5"]["Coordinates"][
                bh_mask_all
            ][bh_mask_ap][iBHmax]
            exclusive_sphere["BHmaxvel"] += data["PartType5"]["Velocities"][
                bh_mask_all
            ][bh_mask_ap][iBHmax]
            exclusive_sphere["BHmaxAR"] += data["PartType5"]["AccretionRates"][
                bh_mask_all
            ][bh_mask_ap][iBHmax]
            exclusive_sphere["BHmaxlasteventa"] += agn_eventa[iBHmax]

        if exclusive_sphere["Mtot"] > 0.0 * exclusive_sphere["Mtot"].units:
            mfrac = mass / exclusive_sphere["Mtot"]
            exclusive_sphere["com"] += (mfrac[:, None] * position).sum(axis=0)
            exclusive_sphere["com"] += centre
            exclusive_sphere["vcom"] += (mfrac[:, None] * velocity).sum(axis=0)
            _, vmax = get_vmax(mass, radius)
            if vmax > 0.0 * vmax.units:
                vrel = velocity - exclusive_sphere["vcom"][None, :]
                Ltot = unyt.array.unorm(
                    (mass[:, None] * unyt.array.ucross(position, vrel)).sum(axis=0)
                )
                exclusive_sphere["spin_parameter"] += Ltot / (
                    np.sqrt(2.0)
                    * exclusive_sphere["Mtot"]
                    * self.physical_radius_mpc
                    * unyt.Mpc
                    * vmax
                )

        if exclusive_sphere["Mgas"] > 0.0 * exclusive_sphere["Mgas"].units:
            frac_mgas = mass_gas / exclusive_sphere["Mgas"]
            vcom_gas = (frac_mgas[:, None] * vel_gas).sum(axis=0)
            Lgas, kappa, Mcountrot = get_angular_momentum_and_kappa_corot(
                mass_gas,
                pos_gas,
                vel_gas,
                ref_velocity=vcom_gas,
                do_counterrot_mass=True,
            )
            exclusive_sphere["Lgas"] += Lgas
            exclusive_sphere["kappa_corot_gas"] += kappa
            exclusive_sphere["DtoTgas"] += (
                1.0 - 2.0 * Mcountrot / exclusive_sphere["Mgas"]
            )

            """
            exclusive_sphere["veldisp_matrix_gas"] += get_velocity_dispersion_matrix(
                frac_mgas, vel_gas, vcom_gas
            )
            """

            # below we need to force conversion to np.float64 before summing
            # up particles to avoid overflow
            ekin_gas = mass_gas * ((vel_gas - vcom_gas) ** 2).sum(axis=1)
            ekin_gas = unyt.unyt_array(
                ekin_gas.value, dtype=np.float64, units=ekin_gas.units
            )
            exclusive_sphere["Ekin_gas"] += 0.5 * ekin_gas.sum()

            exclusive_sphere["GasAxisLengths"] += get_axis_lengths(mass_gas, pos_gas)

        if exclusive_sphere["Mdm"] > 0.0 * exclusive_sphere["Mdm"].units:
            frac_mdm = mass_dm / exclusive_sphere["Mdm"]
            vcom_dm = (frac_mdm[:, None] * vel_dm).sum(axis=0)
            exclusive_sphere["Ldm"] += get_angular_momentum(
                mass_dm, pos_dm, vel_dm, ref_velocity=vcom_dm
            )
            exclusive_sphere["DMAxisLengths"] += get_axis_lengths(mass_dm, pos_dm)

            """
            exclusive_sphere["veldisp_matrix_dm"] += get_velocity_dispersion_matrix(
                frac_mdm, vel_dm, vcom_dm
            )
            """

        if exclusive_sphere["Mstar"] > 0.0 * exclusive_sphere["Mstar"].units:
            frac_mstar = mass_star / exclusive_sphere["Mstar"]
            vcom_star = (frac_mstar[:, None] * vel_star).sum(axis=0)
            Lstar, kappa, Mcountrot = get_angular_momentum_and_kappa_corot(
                mass_star,
                pos_star,
                vel_star,
                ref_velocity=vcom_star,
                do_counterrot_mass=True,
            )
            exclusive_sphere["Lstar"] += Lstar
            exclusive_sphere["kappa_corot_star"] += kappa
            exclusive_sphere["DtoTstar"] += (
                1.0 - 2.0 * Mcountrot / exclusive_sphere["Mstar"]
            )
            exclusive_sphere["StellarAxisLengths"] += get_axis_lengths(
                mass_star, pos_star
            )

            """
            exclusive_sphere["veldisp_matrix_star"] += get_velocity_dispersion_matrix(
                frac_mstar, vel_star, vcom_star
            )
            """

            # below we need to force conversion to np.float64 before summing
            # up particles to avoid overflow
            ekin_star = mass_star * ((vel_star - vcom_star) ** 2).sum(axis=1)
            ekin_star = unyt.unyt_array(
                ekin_star.value, dtype=np.float64, units=ekin_star.units
            )
            exclusive_sphere["Ekin_star"] += 0.5 * ekin_star.sum()

        if (
            exclusive_sphere["Mgas"] + exclusive_sphere["Mstar"]
            > 0.0 * exclusive_sphere["Mgas"].units
        ):
            frac_mbar = mass_baryons / (
                exclusive_sphere["Mgas"] + exclusive_sphere["Mstar"]
            )
            vcom_bar = (frac_mbar[:, None] * vel_baryons).sum(axis=0)
            Lbar, kappa = get_angular_momentum_and_kappa_corot(
                mass_baryons, pos_baryons, vel_baryons, ref_velocity=vcom_bar
            )
            exclusive_sphere["Lbaryons"] += Lbar
            exclusive_sphere["kappa_corot_baryons"] += kappa
            exclusive_sphere["BaryonAxisLengths"] += get_axis_lengths(
                mass_baryons, pos_baryons
            )

        if exclusive_sphere["Ngas"] > 0:
            gas_mask_all = data["PartType0"]["GroupNr_bound"] == index
            SFR = data["PartType0"]["StarFormationRates"][gas_mask_all][gas_mask_ap]
            # negative values of SFR are not SFR at all!
            is_SFR = SFR > 0.0
            exclusive_sphere["SFR"] += SFR[is_SFR].sum()
            exclusive_sphere["Mgas_SF"] += mass_gas[is_SFR].sum()
            Mgasmetal = (
                mass_gas
                * data["PartType0"]["MetalMassFractions"][gas_mask_all][gas_mask_ap]
            )
            exclusive_sphere["Mgasmetal_SF"] += Mgasmetal[is_SFR].sum()
            exclusive_sphere["Mgasmetal"] += Mgasmetal.sum()
            MgasO = (
                mass_gas
                * data["PartType0"]["SmoothedElementMassFractions"][gas_mask_all][
                    gas_mask_ap
                ][:, indexO]
            )
            exclusive_sphere["MgasO_SF"] += MgasO[is_SFR].sum()
            exclusive_sphere["MgasO"] += MgasO.sum()
            MgasFe = (
                mass_gas
                * data["PartType0"]["SmoothedElementMassFractions"][gas_mask_all][
                    gas_mask_ap
                ][:, indexFe]
            )
            exclusive_sphere["MgasFe_SF"] += MgasFe[is_SFR].sum()
            exclusive_sphere["MgasFe"] += MgasFe.sum()
            gas_temp = data["PartType0"]["Temperatures"][gas_mask_all][gas_mask_ap]
            last_agn_gas = data["PartType0"]["LastAGNFeedbackScaleFactors"][
                gas_mask_all
            ][gas_mask_ap]
            no_agn = ~self.filter.is_recently_heated(last_agn_gas, gas_temp)
            exclusive_sphere["Tgas"] += (
                (mass_gas / exclusive_sphere["Mgas"]) * gas_temp
            ).sum()
            if np.any(no_agn):
                mass_gas_no_agn = mass_gas[no_agn]
                Mgas_no_agn = mass_gas_no_agn.sum()
                if Mgas_no_agn > 0.0:
                    exclusive_sphere["Tgas_no_agn"] += (
                        (mass_gas_no_agn / Mgas_no_agn) * gas_temp[no_agn]
                    ).sum()

        for name, r, m, M in zip(
            [
                "HalfMassRadiusGas",
                "HalfMassRadiusDM",
                "HalfMassRadiusStar",
            ],
            [
                radius[type == "PartType0"],
                radius[type == "PartType1"],
                radius[type == "PartType4"],
            ],
            [mass_gas, mass_dm, mass_star],
            [
                exclusive_sphere["Mgas"],
                exclusive_sphere["Mdm"],
                exclusive_sphere["Mstar"],
            ],
        ):
            exclusive_sphere[name] += get_half_mass_radius(r, m, M)
            if exclusive_sphere[name] >= self.physical_radius_mpc * unyt.Mpc:
                raise RuntimeError(
                    "Half mass radius '{name}' larger than aperture"
                    f" ({exclusive_sphere[name]} >="
                    f" {self.physical_radius_mpc * unyt.Mpc}!"
                    " This should not happen."
                )

        prefix = f"ExclusiveSphere/{self.physical_radius_mpc*1000.:.0f}kpc"
        for name, outputname, _, _, _, description, _ in self.property_list:
            halo_result.update(
                {
                    f"{prefix}/{outputname}": (
                        exclusive_sphere[name],
                        description,
                    )
                }
            )

        return


class DummyExclusiveSphereProperties(ExclusiveSphereProperties):
    """
    Dummy ExclusiveSphereProperties object that can be used to test the code.

    The dummy object does not require any input arguments, except for a
    DummyHaloGenerator that can generate a minimal cellgrid that is needed
    for the RecentlyHeatedGasFilter.
    """

    def __init__(self, dummy_halos):

        self.filter = RecentlyHeatedGasFilter(dummy_halos.get_cell_grid())
        self.physical_radius_mpc = 0.05


def test_exclusive_sphere_properties():
    """
    Unit test for the exclusive sphere property calculations.

    We generate 100 random "dummy" halos and feed them to
    ExclusiveSphereProperties::calculate(). We check that the returned values
    are present, and have the right units, size and dtype
    """

    from dummy_halo_generator import DummyHaloGenerator

    # initialise the DummyHaloGenerator with a random seed
    dummy_halos = DummyHaloGenerator(3256)

    # generate a minimal ExclusiveSphereProperties object that does not require
    # an actual snapshot
    property_calculator = DummyExclusiveSphereProperties(dummy_halos)

    # generate 100 random halos
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

        # check that the calculation returns the correct values
        for (
            _,
            outputname,
            size,
            dtype,
            unit_string,
            _,
            _,
        ) in property_calculator.property_list:
            full_name = f"ExclusiveSphere/50kpc/{outputname}"
            assert full_name in halo_result
            result = halo_result[full_name][0]
            assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
            assert result.dtype == dtype
            unit = unyt.Unit(unit_string)
            assert result.units.same_dimensions_as(unit.units)


if __name__ == "__main__":
    """
    Standalone version of the program: just run the unit test.

    Note that this can also be achieved by running "pytest *.py" in the folder.
    """
    print("Running test_exclusive_sphere_properties()...")
    test_exclusive_sphere_properties()
    print("Test passed.")
