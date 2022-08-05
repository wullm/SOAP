#!/bin/env python

import numpy as np
import unyt

from halo_properties import HaloProperty
from dataset_names import mass_dataset
from half_mass_radius import get_half_mass_radius
from kinematic_properties import (
    get_angular_momentum,
    get_angular_momentum_and_kappa_corot,
    get_vmax,
)
from recently_heated_gas_filter import RecentlyHeatedGasFilter


class SubhaloProperties(HaloProperty):

    # List of properties that get computed
    # For each property, we have the following columns:
    #  - name: Name of the property within calculate() and in the output file
    #  - shape: Shape of this property for a single halo (1: scalar, 3: vector...)
    #  - dtype: Data type that will be used. Should have enough precision to avoid over/underflow
    #  - unit: Units that will be used internally and for the output.
    #  - description: Description string that will be used to describe the property in the output.
    property_list = [
        ("Mtot", 1, np.float32, "Msun", "Total mass."),
        ("Mgas", 1, np.float32, "Msun", "Total gas mass."),
        ("Mdm", 1, np.float32, "Msun", "Total DM mass."),
        ("Mstar", 1, np.float32, "Msun", "Total stellar mass."),
        ("Mstar_init", 1, np.float32, "Msun", "Total stellar initial mass."),
        ("Mbh_dynamical", 1, np.float32, "Msun", "Total BH dynamical mass."),
        ("Mbh_subgrid", 1, np.float32, "Msun", "Total BH subgrid mass."),
        ("Ngas", 1, np.uint32, unyt.dimensionless, "Number of gas particles."),
        ("Ndm", 1, np.uint32, unyt.dimensionless, "Number of dark matter particles."),
        ("Nstar", 1, np.uint32, unyt.dimensionless, "Number of star particles."),
        ("Nbh", 1, np.uint32, unyt.dimensionless, "Number of black hole particles."),
        (
            "BHlasteventa",
            1,
            np.float32,
            unyt.dimensionless,
            "Scale-factor of last AGN event.",
        ),
        ("BHmaxM", 1, np.float32, "Msun", "Mass of most massive black hole."),
        ("BHmaxID", 1, np.uint64, unyt.dimensionless, "ID of most massive black hole."),
        ("BHmaxpos", 3, np.float64, unyt.kpc, "Position of most massive black hole."),
        (
            "BHmaxvel",
            3,
            np.float32,
            unyt.km / unyt.s,
            "Velocity of most massive black hole.",
        ),
        (
            "BHmaxAR",
            1,
            np.float32,
            "Msun/yr",
            "Accretion rate of most massive black hole.",
        ),
        (
            "BHmaxlasteventa",
            1,
            np.float32,
            unyt.dimensionless,
            "Scale-factor of last AGN event for most massive black hole.",
        ),
        ("com", 3, np.float32, unyt.kpc, "Centre of mass."),
        ("vcom", 3, np.float32, unyt.km / unyt.s, "Centre of mass velocity."),
        (
            "Lgas",
            3,
            np.float32,
            "Msun*km*kpc/s",
            "Total angular momentum of the gas, relative w.r.t. the centre of potential and gas bulk velocity.",
        ),
        (
            "Ldm",
            3,
            np.float32,
            "Msun*km*kpc/s",
            "Total angular momentum of the dark matter, relative w.r.t. the centre of potential and DM bulk velocity.",
        ),
        (
            "Lstar",
            3,
            np.float32,
            "Msun*km*kpc/s",
            "Total angular momentum of the stars, relative w.r.t. the centre of potential and stellar bulk velocity.",
        ),
        (
            "kappa_corot_gas",
            1,
            np.float32,
            unyt.dimensionless,
            "Kappa corot for gas, relative w.r.t. the centre of potential and the bulk velocity of the gas.",
        ),
        (
            "kappa_corot_star",
            1,
            np.float32,
            unyt.dimensionless,
            "Kappa corot for stars, relative w.r.t. the centre of potential and the bulk velocity of the stars.",
        ),
        (
            "Lbaryons",
            3,
            np.float32,
            "Msun*km*kpc/s",
            "Total angular momentum of baryons (gas and stars), relative w.r.t. the centre of potential and baryonic bulk velocity.",
        ),
        (
            "kappa_corot_baryons",
            1,
            np.float32,
            unyt.dimensionless,
            "Kappa corot for baryons (gas and stars), relative w.r.t. the centre of potential and the bulk velocity of the baryons.",
        ),
        ("Mgasmetal", 1, np.float32, "Msun", "Total gas mass in metals."),
        ("Tgas", 1, np.float32, unyt.K, "Mass-weighted gas temperature."),
        (
            "Tgas_no_cool",
            1,
            np.float32,
            unyt.K,
            "Mass-weighted gas temperature, excluding cool (T<1e5 K) gas.",
        ),
        (
            "Tgas_no_agn",
            1,
            np.float32,
            unyt.K,
            "Mass-weighted gas temperature, excluding gas that was recently heated by AGN.",
        ),
        (
            "Tgas_no_cool_no_agn",
            1,
            np.float32,
            unyt.K,
            "Mass-weighted gas temperature, excluding cool (T<1e5 K) gas and gas that was heated by AGN less than 15 Myr ago.",
        ),
        ("SFR", 1, np.float32, unyt.Msun / unyt.yr, "Total SFR."),
        (
            "StellarLuminosity",
            9,
            np.float32,
            unyt.dimensionless,
            "Total stellar luminosity in the 9 GAMA bands.",
        ),
        ("Mstarmetal", 1, np.float32, "Msun", "Total stellar mass in metals."),
        ("Vmax", 1, np.float32, unyt.km / unyt.s, "Maximum circular velocity."),
        ("R_vmax", 1, np.float32, unyt.kpc, "Radius at which Vmax is reached."),
        (
            "spin_parameter",
            1,
            np.float32,
            unyt.dimensionless,
            "Bullock et al. (2001) spin parameter.",
        ),
        ("HalfMassRadiusTot", 1, np.float32, unyt.kpc, "Total half mass radius."),
        ("HalfMassRadiusGas", 1, np.float32, unyt.kpc, "Total gas half mass radius."),
        ("HalfMassRadiusDM", 1, np.float32, unyt.kpc, "Total DM half mass radius."),
        (
            "HalfMassRadiusStar",
            1,
            np.float32,
            unyt.kpc,
            "Total stellar half mass radius.",
        ),
    ]

    def __init__(self, cellgrid, recently_heated_gas_filter, bound_only=True):
        super().__init__(cellgrid)

        self.bound_only = bound_only
        self.filter = recently_heated_gas_filter

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
            "PartType0": [
                "Coordinates",
                "LastAGNFeedbackScaleFactors",
                "Masses",
                "MetalMassFractions",
                "StarFormationRates",
                "Temperatures",
                "Velocities",
                self.grnr,
            ],
            "PartType1": ["Coordinates", "Masses", "Velocities", self.grnr],
            "PartType4": [
                "Coordinates",
                "InitialMasses",
                "Luminosities",
                "Masses",
                "MetalMassFractions",
                "Velocities",
                self.grnr,
            ],
            "PartType5": [
                "AccretionRates",
                "Coordinates",
                "DynamicalMasses",
                "LastAGNFeedbackScaleFactors",
                "ParticleIDs",
                "SubgridMasses",
                "Velocities",
                self.grnr,
            ],
        }

    def calculate(self, input_halo, search_radius, data, halo_result):
        """
        Compute centre of mass etc of bound particles

        input_halo       - dict with halo properties passed in from VR (see
                           halo_centres.py)
        search_radius    - radius in which we have all particles
        data             - contains particle data. E.g. data["PartType1"]["Coordinates"]
                           has the particle coordinates for type 1
        halo_result      - dict with halo properties computed so far. Properties
                           computed here should be added to halo_result.

        Input particle data arrays are unyt_arrays.
        """

        # Look up centre and array index of this halo in VR catalogue
        centre = input_halo["cofp"]
        index = input_halo["index"]

        types_present = [type for type in self.particle_properties if type in data]

        mass = []
        position = []
        radius = []
        velocity = []
        types = []
        for ptype in types_present:
            grnr = data[ptype][self.grnr]
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

        subhalo = {}
        # declare all the variables we will compute
        # we set them to 0 in case a particular variable cannot be computed
        # all variables are defined with physical units and an appropriate dtype
        # we need to use the custom unit registry so that everything can be converted
        # back to snapshot units in the end
        for name, shape, dtype, unit, _ in self.property_list:
            if shape > 1:
                val = [0] * shape
            else:
                val = 0
            subhalo[name] = unyt.unyt_array(
                val, dtype=dtype, units=unit, registry=mass.units.registry
            )

        gas_mask_sh = types == "PartType0"
        dm_mask_sh = types == "PartType1"
        star_mask_sh = types == "PartType4"
        bh_mask_sh = types == "PartType5"
        baryons_mask_sh = (types == "PartType0") | (types == "PartType4")

        subhalo["Ngas"] = (
            gas_mask_sh.sum(dtype=subhalo["Ngas"].dtype) * subhalo["Ngas"].units
        )
        subhalo["Ndm"] = (
            dm_mask_sh.sum(dtype=subhalo["Ndm"].dtype) * subhalo["Ndm"].units
        )
        subhalo["Nstar"] = (
            star_mask_sh.sum(dtype=subhalo["Nstar"].dtype) * subhalo["Nstar"].units
        )
        subhalo["Nbh"] = (
            bh_mask_sh.sum(dtype=subhalo["Nbh"].dtype) * subhalo["Nbh"].units
        )

        mass_gas = mass[gas_mask_sh]
        mass_dm = mass[dm_mask_sh]
        mass_star = mass[star_mask_sh]
        mass_baryons = mass[baryons_mask_sh]

        pos_gas = position[gas_mask_sh]
        pos_dm = position[dm_mask_sh]
        pos_star = position[star_mask_sh]
        pos_baryons = position[baryons_mask_sh]

        vel_gas = velocity[gas_mask_sh]
        vel_dm = velocity[dm_mask_sh]
        vel_star = velocity[star_mask_sh]
        vel_baryons = velocity[baryons_mask_sh]

        subhalo["Mtot"] += mass.sum()
        subhalo["Mgas"] += mass_gas.sum()
        subhalo["Mdm"] = mass_dm.sum()
        subhalo["Mstar"] += mass_star.sum()
        subhalo["Mbh_dynamical"] += mass[bh_mask_sh].sum()

        if subhalo["Nstar"] > 0:
            star_mask_all = data["PartType4"][self.grnr] == index
            subhalo["Mstar_init"] += data["PartType4"]["InitialMasses"][
                star_mask_all
            ].sum()
            subhalo["StellarLuminosity"] += data["PartType4"]["Luminosities"][
                star_mask_all
            ].sum(axis=0)
            subhalo["Mstarmetal"] += (
                mass_star * data["PartType4"]["MetalMassFractions"][star_mask_all]
            ).sum()

        if subhalo["Nbh"] > 0:
            bh_mask_all = data["PartType5"][self.grnr] == index
            subhalo["Mbh_subgrid"] += data["PartType5"]["SubgridMasses"][
                bh_mask_all
            ].sum()

            agn_eventa = data["PartType5"]["LastAGNFeedbackScaleFactors"][bh_mask_all]

            subhalo["BHlasteventa"] += np.max(agn_eventa)

            iBHmax = np.argmax(data["PartType5"]["SubgridMasses"][bh_mask_all])
            subhalo["BHmaxM"] += data["PartType5"]["SubgridMasses"][bh_mask_all][iBHmax]
            subhalo["BHmaxID"] = (
                data["PartType5"]["ParticleIDs"][bh_mask_all][iBHmax].astype(
                    subhalo["BHmaxID"].dtype
                )
                * subhalo["BHmaxID"].units
            )
            subhalo["BHmaxpos"] += data["PartType5"]["Coordinates"][bh_mask_all][iBHmax]
            subhalo["BHmaxvel"] += data["PartType5"]["Velocities"][bh_mask_all][iBHmax]
            subhalo["BHmaxAR"] += data["PartType5"]["AccretionRates"][bh_mask_all][
                iBHmax
            ]
            subhalo["BHmaxlasteventa"] += agn_eventa[iBHmax]

        if subhalo["Mtot"] > 0.0 * subhalo["Mtot"].units:
            mfrac = mass / subhalo["Mtot"]
            subhalo["com"] += (mfrac[:, None] * position).sum(axis=0)
            subhalo["com"] += centre
            subhalo["vcom"] += (mfrac[:, None] * velocity).sum(axis=0)
            r_vmax, vmax = get_vmax(mass, radius)
            subhalo["R_vmax"] += r_vmax
            subhalo["Vmax"] += vmax
            if r_vmax > 0.0 * r_vmax.units and vmax > 0.0 * vmax.units:
                mask_r_vmax = radius <= r_vmax
                vrel = velocity[mask_r_vmax, :] - subhalo["vcom"][None, :]
                Ltot = unyt.array.unorm(
                    (
                        mass[mask_r_vmax, None]
                        * unyt.array.ucross(position[mask_r_vmax, :], vrel)
                    ).sum(axis=0)
                )
                M_r_vmax = mass[mask_r_vmax].sum()
                if M_r_vmax > 0.0 * M_r_vmax.units:
                    subhalo["spin_parameter"] += Ltot / (
                        np.sqrt(2.0) * M_r_vmax * vmax * r_vmax
                    )

        if subhalo["Mgas"] > 0.0 * subhalo["Mgas"].units:
            frac_mgas = mass_gas / subhalo["Mgas"]
            vcom_gas = (frac_mgas[:, None] * vel_gas).sum(axis=0)
            Lgas, kappa = get_angular_momentum_and_kappa_corot(
                mass_gas, pos_gas, vel_gas, ref_velocity=vcom_gas
            )
            subhalo["Lgas"] += Lgas
            subhalo["kappa_corot_gas"] += kappa

        if subhalo["Mdm"] > 0.0 * subhalo["Mdm"].units:
            frac_mdm = mass_dm / subhalo["Mdm"]
            vcom_dm = (frac_mdm[:, None] * vel_dm).sum(axis=0)
            subhalo["Ldm"] += get_angular_momentum(
                mass_dm, pos_dm, vel_dm, ref_velocity=vcom_dm
            )

        if subhalo["Mstar"] > 0.0 * subhalo["Mstar"].units:
            frac_mstar = mass_star / subhalo["Mstar"]
            vcom_star = (frac_mstar[:, None] * vel_star).sum(axis=0)
            Lstar, kappa = get_angular_momentum_and_kappa_corot(
                mass_star, pos_star, vel_star, ref_velocity=vcom_star
            )
            subhalo["Lstar"] += Lstar
            subhalo["kappa_corot_star"] += kappa

        if subhalo["Mgas"] + subhalo["Mstar"] > 0.0 * subhalo["Mgas"].units:
            frac_mbar = mass_baryons / (subhalo["Mgas"] + subhalo["Mstar"])
            vcom_bar = (frac_mbar[:, None] * vel_baryons).sum(axis=0)
            Lbar, kappa = get_angular_momentum_and_kappa_corot(
                mass_baryons, pos_baryons, vel_baryons, ref_velocity=vcom_bar
            )
            subhalo["Lbaryons"] += Lbar
            subhalo["kappa_corot_baryons"] += kappa

        if subhalo["Ngas"] > 0:
            gas_mask_all = data["PartType0"][self.grnr] == index
            SFR = data["PartType0"]["StarFormationRates"][gas_mask_all]
            # negative values of SFR are not SFR at all!
            is_SFR = SFR > 0.0
            subhalo["SFR"] += SFR[is_SFR].sum()
            Mgasmetal = mass_gas * data["PartType0"]["MetalMassFractions"][gas_mask_all]
            subhalo["Mgasmetal"] += Mgasmetal.sum()
            gas_temp = data["PartType0"]["Temperatures"][gas_mask_all]
            last_agn_gas = data["PartType0"]["LastAGNFeedbackScaleFactors"][
                gas_mask_all
            ]
            no_agn = ~self.filter.is_recently_heated(last_agn_gas, gas_temp)
            no_cool = gas_temp >= 1.0e5 * unyt.K
            subhalo["Tgas"] += ((mass_gas / subhalo["Mgas"]) * gas_temp).sum()
            if np.any(no_cool):
                mass_gas_no_cool = mass_gas[no_cool]
                Mgas_no_cool = mass_gas_no_cool.sum()
                if Mgas_no_cool > 0.0 * Mgas_no_cool.units:
                    subhalo["Tgas_no_cool"] += (
                        (mass_gas_no_cool / Mgas_no_cool) * gas_temp[no_cool]
                    ).sum()
            if np.any(no_agn):
                mass_gas_no_agn = mass_gas[no_agn]
                Mgas_no_agn = mass_gas_no_agn.sum()
                if Mgas_no_agn > 0.0 * Mgas_no_agn.units:
                    subhalo["Tgas_no_agn"] += (
                        (mass_gas_no_agn / Mgas_no_agn) * gas_temp[no_agn]
                    ).sum()
            no_cool_no_agn = no_agn & no_cool
            if np.any(no_cool_no_agn):
                mass_gas_no_cool_no_agn = mass_gas[no_cool_no_agn]
                Mgas_no_cool_no_agn = mass_gas_no_cool_no_agn.sum()
                if Mgas_no_cool_no_agn > 0.0 * Mgas_no_cool_no_agn.units:
                    subhalo["Tgas_no_cool_no_agn"] += (
                        (mass_gas_no_cool_no_agn / Mgas_no_cool_no_agn)
                        * gas_temp[no_cool_no_agn]
                    ).sum()

        for name, r, m, M in zip(
            [
                "HalfMassRadiusTot",
                "HalfMassRadiusGas",
                "HalfMassRadiusDM",
                "HalfMassRadiusStar",
            ],
            [
                radius,
                radius[gas_mask_sh],
                radius[dm_mask_sh],
                radius[star_mask_sh],
            ],
            [mass, mass_gas, mass_dm, mass_star],
            [
                subhalo["Mtot"],
                subhalo["Mgas"],
                subhalo["Mdm"],
                subhalo["Mstar"],
            ],
        ):
            subhalo[name] += get_half_mass_radius(r, m, M)

        # Add these properties to the output
        if self.bound_only:
            prefix = "BoundSubhaloProperties"
        else:
            prefix = "FOFSubhaloProperties"
        for name, _, _, _, description in self.property_list:
            halo_result.update(
                {
                    f"{prefix}/{name}": (
                        subhalo[name],
                        description,
                    )
                }
            )


def test_subhalo_properties():
    """
    Unit test for the subhalo property calculations.

    We generate 100 random "dummy" halos and feed them to
    SubhaloProperties::calculate(). We check that the returned values
    are present, and have the right units, size and dtype
    """

    from dummy_halo_generator import DummyHaloGenerator

    # initialise the DummyHaloGenerator with a random seed
    dummy_halos = DummyHaloGenerator(16902)

    recently_heated_gas_filter = RecentlyHeatedGasFilter(dummy_halos.get_cell_grid())

    property_calculator_bound = SubhaloProperties(
        dummy_halos.get_cell_grid(), recently_heated_gas_filter
    )
    property_calculator_both = SubhaloProperties(
        dummy_halos.get_cell_grid(), recently_heated_gas_filter, False
    )

    # generate 100 random halos
    for i in range(100):
        input_halo, data, _, _, _ = dummy_halos.get_random_halo(
            [1, 10, 100, 1000, 10000]
        )

        halo_result = {}
        for subhalo_name, prop_calc in [
            ("BoundSubhaloProperties", property_calculator_bound),
            ("FOFSubhaloProperties", property_calculator_both),
        ]:
            input_data = {}
            for ptype in prop_calc.particle_properties:
                if ptype in data:
                    input_data[ptype] = {}
                    for dset in prop_calc.particle_properties[ptype]:
                        input_data[ptype][dset] = data[ptype][dset]
            input_halo_copy = input_halo.copy()
            input_data_copy = input_data.copy()
            prop_calc.calculate(input_halo, 0.0 * unyt.kpc, input_data, halo_result)
            assert input_halo == input_halo_copy
            assert input_data == input_data_copy

            # check that the calculation returns the correct values
            for (
                name,
                size,
                dtype,
                unit_string,
                _,
            ) in prop_calc.property_list:
                full_name = f"{subhalo_name}/{name}"
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
    print("Running test_subhalo_properties()...")
    test_subhalo_properties()
    print("Test passed.")

    print("Name & Size & Unit & Type & Description \\\\")
    for (
        name,
        size,
        dtype,
        unit,
        description,
    ) in SubhaloProperties.property_list:
        unit_str = unit.__str__()
        unit_str = unit_str.replace("1.98841586e+30 kg", "M$_\\odot{}$")
        print(
            f"\\verb+{name}+ & {size} & {unit_str} & {dtype.__name__} & {description} \\\\"
        )
