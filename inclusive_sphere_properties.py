#!/bin/env python

import numpy as np
import unyt
from scipy.optimize import brentq

from halo_properties import HaloProperty, ReadRadiusTooSmallError
from kinematic_properties import (
    get_velocity_dispersion_matrix,
    get_angular_momentum,
)
from recently_heated_gas_filter import RecentlyHeatedGasFilter
from property_table import PropertyTable

from dataset_names import mass_dataset

from mpi4py import MPI


class InclusiveSphereProperties(HaloProperty):

    # Arrays which must be read in for this calculation.
    # Note that if there are no particles of a given type in the
    # snapshot, that type will not be read in and will not have
    # an entry in the data argument to calculate(), below.
    # (E.g. gas, star or BH particles in DMO runs)
    particle_properties = {
        "PartType0": [
            "ComptonYParameters",
            "Coordinates",
            "Densities",
            "GroupNr_bound",
            "LastAGNFeedbackScaleFactors",
            "Masses",
            "MetalMassFractions",
            "Pressures",
            "Temperatures",
            "Velocities",
            "XrayLuminosities",
            "XrayPhotonLuminosities",
        ],
        "PartType1": ["Coordinates", "GroupNr_bound", "Masses", "Velocities"],
        "PartType4": [
            "Coordinates",
            "GroupNr_bound",
            "InitialMasses",
            "Luminosities",
            "Masses",
            "MetalMassFractions",
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
        PropertyTable.full_property_list[prop]
        for prop in [
            "Mtot",
            "Ngas",
            "Ndm",
            "Nstar",
            "Nbh",
            "com",
            "vcom",
            "Mfrac_satellites",
            "Mgas",
            "Lgas",
            "com_gas",
            "vcom_gas",
            "veldisp_matrix_gas",
            "Mgasmetal",
            "Mhotgas",
            "Tgas_no_cool",
            "Xraylum",
            "Xrayphlum",
            "compY",
            "Xraylum_no_agn",
            "Xrayphlum_no_agn",
            "compY_no_agn",
            "Ekin_gas",
            "Etherm_gas",
            "Mdm",
            "Ldm",
            "veldisp_matrix_dm",
            "Mstar",
            "com_star",
            "vcom_star",
            "veldisp_matrix_star",
            "Lstar",
            "Mstar_init",
            "Mstarmetal",
            "StellarLuminosity",
            "Ekin_star",
            "Lbaryons",
            "Mbh_dynamical",
            "Mbh_subgrid",
            "BHlasteventa",
            "BHmaxM",
            "BHmaxID",
            "BHmaxpos",
            "BHmaxvel",
            "BHmaxAR",
            "BHmaxlasteventa",
        ]
    ]

    def __init__(self, cellgrid, physical_radius_kpc, recently_heated_gas_filter):
        super().__init__(cellgrid)

        self.filter = recently_heated_gas_filter

        self.mean_density_multiple = None
        self.critical_density_multiple = None
        self.physical_radius_mpc = 0.001 * physical_radius_kpc

        self.name = f"inclusive_sphere_{physical_radius_kpc:.0f}kpc"

    def calculate(self, input_halo, search_radius, data, halo_result):
        """
        Compute spherical masses and overdensities for a halo

        input_halo       - dict with halo properties passed in from VR (see
                           halo_centres.py)
        search_radius    - radius in which we have all particles
        data             - contains particle data. E.g. data["PartType1"]["Coordinates"]
                           has the particle coordinates for type 1
        halo_result      - dict with halo properties computed so far. Properties
                           computed here should be added to halo_result.

        Input particle data arrays are unyt_arrays.
        """

        # Find the halo centre of potential
        centre = input_halo["cofp"]

        reg = centre.units.registry

        inclusive_sphere = {}
        # declare all the variables we will compute
        # we set them to 0 in case a particular variable cannot be computed
        # all variables are defined with physical units and an appropriate dtype
        # we need to use the custom unit registry so that everything can be converted
        # back to snapshot units in the end
        for name, shape, dtype, unit, _, _ in self.property_list:
            if shape > 1:
                val = [0] * shape
            else:
                val = 0
            inclusive_sphere[name] = unyt.unyt_array(
                val, dtype=dtype, units=unit, registry=reg
            )

        # inclusive spheres only exist for central galaxies?
        if input_halo["Structuretype"] != 10:
            for name, _, _, _, description, _ in self.property_list:
                halo_result.update(
                    {
                        f"InclusiveSphere/{self.physical_radius_mpc*1000.:.0f}kpc/{name}": (
                            inclusive_sphere[name],
                            description,
                        )
                    }
                )
            return

        # Make an array of particle masses, radii and positions
        mass = []
        radius = []
        position = []
        velocity = []
        types = []
        groupnr = []
        for ptype in data:
            mass.append(data[ptype][mass_dataset(ptype)])
            pos = data[ptype]["Coordinates"] - centre[None, :]
            position.append(pos)
            r = np.sqrt(np.sum(pos**2, axis=1))
            radius.append(r)
            velocity.append(data[ptype]["Velocities"])
            typearr = np.zeros(r.shape, dtype="U9")
            typearr[:] = ptype
            types.append(typearr)
            groupnr.append(data[ptype]["GroupNr_bound"])
        mass = unyt.array.uconcatenate(mass)
        radius = unyt.array.uconcatenate(radius)
        position = unyt.array.uconcatenate(position)
        velocity = unyt.array.uconcatenate(velocity)
        types = np.concatenate(types)
        groupnr = unyt.array.uconcatenate(groupnr)

        # figure out which particles in the list are bound to a halo that is not the
        # central halo
        is_bound_to_satellite = (groupnr >= 0) & (groupnr != input_halo["index"])

        inclusive_sphere["Mtot"] += mass.sum()
        sphere_radius = self.physical_radius_mpc * unyt.Mpc
        if inclusive_sphere["Mtot"] > 0.0 * mass.units:

            gas_selection = radius[types == "PartType0"] < sphere_radius
            dm_selection = radius[types == "PartType1"] < sphere_radius
            star_selection = radius[types == "PartType4"] < sphere_radius
            bh_selection = radius[types == "PartType5"] < sphere_radius

            all_selection = radius < sphere_radius
            mass = mass[all_selection]
            position = position[all_selection]
            velocity = velocity[all_selection]
            types = types[all_selection]
            is_bound_to_satellite = is_bound_to_satellite[all_selection]

            # note that we cannot divide by mSO here, since that was based on an interpolation
            mass_frac = mass / mass.sum()
            inclusive_sphere["com"] += (mass_frac[:, None] * position).sum(axis=0)
            inclusive_sphere["com"] += centre
            inclusive_sphere["vcom"] += (mass_frac[:, None] * velocity).sum(axis=0)

            inclusive_sphere["Mfrac_satellites"] += (
                mass[is_bound_to_satellite].sum() / inclusive_sphere["Mtot"]
            )

            gas_masses = mass[types == "PartType0"]
            gas_pos = position[types == "PartType0"]
            gas_vel = velocity[types == "PartType0"]
            inclusive_sphere["Mgas"] += gas_masses.sum()
            if inclusive_sphere["Mgas"] > 0.0 * inclusive_sphere["Mgas"].units:
                frac_mgas = gas_masses / inclusive_sphere["Mgas"]
                inclusive_sphere["com_gas"] += (frac_mgas[:, None] * gas_pos).sum(
                    axis=0
                )
                inclusive_sphere["com_gas"] += centre
                inclusive_sphere["vcom_gas"] += (frac_mgas[:, None] * gas_vel).sum(
                    axis=0
                )

                inclusive_sphere["Lgas"] += get_angular_momentum(
                    gas_masses,
                    gas_pos,
                    gas_vel,
                    ref_velocity=inclusive_sphere["vcom_gas"],
                )
                inclusive_sphere[
                    "veldisp_matrix_gas"
                ] += get_velocity_dispersion_matrix(
                    frac_mgas, gas_vel, inclusive_sphere["vcom_gas"]
                )

            dm_masses = mass[types == "PartType1"]
            dm_pos = position[types == "PartType1"]
            dm_vel = velocity[types == "PartType1"]
            inclusive_sphere["Mdm"] += dm_masses.sum()
            if inclusive_sphere["Mdm"] > 0.0 * inclusive_sphere["Mdm"].units:
                frac_mdm = dm_masses / inclusive_sphere["Mdm"]
                vcom_dm = (frac_mdm[:, None] * dm_vel).sum(axis=0)

                inclusive_sphere["Ldm"] += get_angular_momentum(
                    dm_masses, dm_pos, dm_vel, ref_velocity=vcom_dm
                )
                inclusive_sphere["veldisp_matrix_dm"] += get_velocity_dispersion_matrix(
                    frac_mdm, dm_vel, vcom_dm
                )

            star_masses = mass[types == "PartType4"]
            star_pos = position[types == "PartType4"]
            star_vel = velocity[types == "PartType4"]
            inclusive_sphere["Mstar"] += star_masses.sum()
            if inclusive_sphere["Mstar"] > 0.0 * inclusive_sphere["Mstar"].units:
                frac_mstar = star_masses / inclusive_sphere["Mstar"]
                inclusive_sphere["com_star"] += (frac_mstar[:, None] * star_pos).sum(
                    axis=0
                )
                inclusive_sphere["com_star"] += centre
                inclusive_sphere["vcom_star"] += (frac_mstar[:, None] * star_vel).sum(
                    axis=0
                )

                inclusive_sphere["Lstar"] += get_angular_momentum(
                    star_masses,
                    star_pos,
                    star_vel,
                    ref_velocity=inclusive_sphere["vcom_star"],
                )
                inclusive_sphere[
                    "veldisp_matrix_star"
                ] += get_velocity_dispersion_matrix(
                    frac_mstar, star_vel, inclusive_sphere["vcom_star"]
                )

            baryon_masses = mass[(types == "PartType0") | (types == "PartType4")]
            baryon_pos = position[(types == "PartType0") | (types == "PartType4")]
            baryon_vel = velocity[(types == "PartType0") | (types == "PartType4")]
            Mbaryons = baryon_masses.sum()
            if Mbaryons > 0.0 * Mbaryons.units:
                baryon_vcom = ((baryon_masses / Mbaryons)[:, None] * baryon_vel).sum(
                    axis=0
                )
                baryon_relvel = baryon_vel - baryon_vcom[None, :]
                inclusive_sphere["Lbaryons"] += (
                    baryon_masses[:, None]
                    * unyt.array.ucross(baryon_pos, baryon_relvel)
                ).sum(axis=0)

            inclusive_sphere["Mbh_dynamical"] += mass[types == "PartType5"].sum()

            # gas specific properties. We (can) only do these if we have gas.
            # (remember that "PartType0" might not be part of 'data' at all)
            if np.any(gas_selection):
                inclusive_sphere["Ngas"] = (
                    gas_selection.sum(dtype=inclusive_sphere["Ngas"].dtype)
                    * inclusive_sphere["Ngas"].units
                )

                inclusive_sphere["Mgasmetal"] += (
                    gas_masses * data["PartType0"]["MetalMassFractions"][gas_selection]
                ).sum()

                gas_temperatures = data["PartType0"]["Temperatures"][gas_selection]
                Tgas_selection = gas_temperatures > 1.0e5 * unyt.K
                inclusive_sphere["Mhotgas"] += gas_masses[Tgas_selection].sum()

                if np.any(Tgas_selection):
                    inclusive_sphere["Tgas_no_cool"] += (
                        gas_temperatures[Tgas_selection] * gas_masses[Tgas_selection]
                    ).sum() / inclusive_sphere["Mhotgas"]

                xraylum = data["PartType0"]["XrayLuminosities"][gas_selection]
                xrayphlum = data["PartType0"]["XrayPhotonLuminosities"][gas_selection]
                inclusive_sphere["Xraylum"] += xraylum.sum()
                inclusive_sphere["Xrayphlum"] += xrayphlum.sum()

                compY = data["PartType0"]["ComptonYParameters"][gas_selection]
                # unyt has some internal issue that causes an overflow when
                # converting from compY.units to SO["compY"].units.
                # we avoid this issue by manually converting the unit
                unit = 1.0 * compY.units
                new_unit = unit.to(inclusive_sphere["compY"].units)
                inclusive_sphere["compY"] += compY.sum().value * new_unit

                last_agn_gas = data["PartType0"]["LastAGNFeedbackScaleFactors"][
                    gas_selection
                ]
                no_agn = ~self.filter.is_recently_heated(last_agn_gas, gas_temperatures)
                if np.any(no_agn):
                    inclusive_sphere["Xraylum_no_agn"] += xraylum[no_agn].sum()
                    inclusive_sphere["Xrayphlum_no_agn"] += xrayphlum[no_agn].sum()
                    inclusive_sphere["compY_no_agn"] += (
                        compY[no_agn].sum().value * new_unit
                    )

                # below we need to force conversion to np.float64 before summing up particles
                # to avoid overflow
                ekin_gas = gas_masses * (
                    (
                        velocity[types == "PartType0"]
                        - inclusive_sphere["vcom_gas"][None, :]
                    )
                    ** 2
                ).sum(axis=1)
                ekin_gas = unyt.unyt_array(
                    ekin_gas.value, dtype=np.float64, units=ekin_gas.units
                )
                inclusive_sphere["Ekin_gas"] += 0.5 * ekin_gas.sum()
                etherm_gas = (
                    1.5
                    * gas_masses
                    * data["PartType0"]["Pressures"][gas_selection]
                    / data["PartType0"]["Densities"][gas_selection]
                )
                etherm_gas = unyt.unyt_array(
                    etherm_gas.value, dtype=np.float64, units=etherm_gas.units
                )
                inclusive_sphere["Etherm_gas"] += etherm_gas.sum()

            if np.any(dm_selection):
                inclusive_sphere["Ndm"] = (
                    dm_selection.sum(dtype=inclusive_sphere["Ndm"].dtype)
                    * inclusive_sphere["Ndm"].units
                )

            # star specific properties
            if np.any(star_selection):
                inclusive_sphere["Nstar"] = (
                    star_selection.sum(dtype=inclusive_sphere["Nstar"].dtype)
                    * inclusive_sphere["Nstar"].units
                )

                inclusive_sphere["Mstar_init"] += data["PartType4"]["InitialMasses"][
                    star_selection
                ].sum()
                inclusive_sphere["Mstarmetal"] += (
                    star_masses
                    * data["PartType4"]["MetalMassFractions"][star_selection]
                ).sum()
                inclusive_sphere["StellarLuminosity"] += data["PartType4"][
                    "Luminosities"
                ][star_selection].sum()

                # below we need to force conversion to np.float64 before summing up particles
                # to avoid overflow
                ekin_star = star_masses * (
                    (
                        velocity[types == "PartType4"]
                        - inclusive_sphere["vcom_star"][None, :]
                    )
                    ** 2
                ).sum(axis=1)
                ekin_star = unyt.unyt_array(
                    ekin_star.value, dtype=np.float64, units=ekin_star.units
                )
                inclusive_sphere["Ekin_star"] += 0.5 * ekin_star.sum()

            # BH specific properties
            if np.any(bh_selection):
                inclusive_sphere["Nbh"] = (
                    bh_selection.sum(dtype=inclusive_sphere["Nbh"].dtype)
                    * inclusive_sphere["Nbh"].units
                )

                inclusive_sphere["Mbh_subgrid"] += data["PartType5"]["SubgridMasses"][
                    bh_selection
                ].sum()
                agn_eventa = data["PartType5"]["LastAGNFeedbackScaleFactors"][
                    bh_selection
                ]

                inclusive_sphere["BHlasteventa"] += np.max(agn_eventa)

                iBHmax = np.argmax(data["PartType5"]["SubgridMasses"][bh_selection])
                inclusive_sphere["BHmaxM"] += data["PartType5"]["SubgridMasses"][
                    bh_selection
                ][iBHmax]
                # unyt annoyingly converts to a floating point type if you use '+='
                # the only way to avoid this is by directly setting the data for the unyt_array
                # however, that is unsafe and results in a warning
                # the only option left is to replace the array with a new copy
                inclusive_sphere["BHmaxID"] = unyt.unyt_array(
                    data["PartType5"]["ParticleIDs"][bh_selection][iBHmax].value,
                    dtype=inclusive_sphere["BHmaxID"].dtype,
                    units=inclusive_sphere["BHmaxID"].units,
                )
                inclusive_sphere["BHmaxpos"] += data["PartType5"]["Coordinates"][
                    bh_selection
                ][iBHmax]
                inclusive_sphere["BHmaxvel"] += data["PartType5"]["Velocities"][
                    bh_selection
                ][iBHmax]
                inclusive_sphere["BHmaxAR"] += data["PartType5"]["AccretionRates"][
                    bh_selection
                ][iBHmax]
                inclusive_sphere["BHmaxlasteventa"] += agn_eventa[iBHmax]

        # Return value should be a dict containing unyt_arrays and descriptions.
        # The dict keys will be used as HDF5 dataset names in the output.
        for name, _, _, _, description, _ in self.property_list:
            halo_result.update(
                {
                    f"InclusiveSphere/{self.physical_radius_mpc*1000.:.0f}kpc/{name}": (
                        inclusive_sphere[name],
                        description,
                    )
                }
            )

        return


def test_inclusive_sphere_properties():

    from dummy_halo_generator import DummyHaloGenerator

    dummy_halos = DummyHaloGenerator(6003)
    filter = RecentlyHeatedGasFilter(dummy_halos.get_cell_grid())

    property_calculator_50kpc = InclusiveSphereProperties(
        dummy_halos.get_cell_grid(), 50.0, filter
    )

    for i in range(100):
        input_halo, data, rmax, Mtot, Npart = dummy_halos.get_random_halo(
            [2, 10, 100, 1000, 10000], has_neutrinos=True
        )

        halo_result = {}
        input_data = {}
        for ptype in property_calculator_50kpc.particle_properties:
            if ptype in data:
                input_data[ptype] = {}
                for dset in property_calculator_50kpc.particle_properties[ptype]:
                    input_data[ptype][dset] = data[ptype][dset]
        input_halo_copy = input_halo.copy()
        input_data_copy = input_data.copy()
        property_calculator_50kpc.calculate(input_halo, rmax, input_data, halo_result)
        # make sure the calculation does not change the input
        assert input_halo_copy == input_halo
        assert input_data_copy == input_data

        for (
            name,
            size,
            dtype,
            unit_string,
            _,
            _,
        ) in property_calculator_50kpc.property_list:
            full_name = f"InclusiveSphere/50kpc/{name}"
            assert full_name in halo_result
            result = halo_result[full_name][0]
            assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
            assert result.dtype == dtype
            unit = unyt.Unit(unit_string)
            assert result.units.same_dimensions_as(unit.units)


if __name__ == "__main__":
    """
    Standalone mode. Just run test_inclusive_sphere_properties().
    """
    print("Calling test_inclusive_sphere_properties()...")
    test_inclusive_sphere_properties()
    print("Test passed.")
