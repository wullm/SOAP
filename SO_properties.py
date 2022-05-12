#!/bin/env python

import numpy as np
import unyt
from halo_properties import HaloProperty, ReadRadiusTooSmallError

from dataset_names import mass_dataset


class SOProperties(HaloProperty):

    # Arrays which must be read in for this calculation.
    # Note that if there are no particles of a given type in the
    # snapshot, that type will not be read in and will not have
    # an entry in the data argument to calculate(), below.
    # (E.g. gas, star or BH particles in DMO runs)
    particle_properties = {
        "PartType0": [
            "ComptonYParameters",
            "Coordinates",
            "Masses",
            "MetalMassFractions",
            "Temperatures",
            "Velocities",
            "XrayLuminosities",
            "XrayPhotonLuminosities",
        ],
        "PartType1": ["Coordinates", "Masses", "Velocities"],
        "PartType4": [
            "Coordinates",
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
            "LastAGNFeedbackScaleFactors",
            "ParticleIDs",
            "SubgridMasses",
            "Velocities",
        ],
    }

    # List of properties that get computed
    # For each property, we have the following columns:
    #  - name: Name of the property within calculate() and in the output file
    #  - shape: Shape of this property for a single halo (1: scalar, 3: vector...)
    #  - dtype: Data type that will be used. Should have enough precision to avoid over/underflow
    #  - unit: Units that will be used internally and for the output.
    #  - description: Description string that will be used to describe the property in the output.
    #                 Should contain a "{label}" entry that will be adjusted to describe the sphere for this
    #                 particular type of SO.
    SO_properties = [
        # global properties
        ("r", 1, np.float32, "Mpc", "Radius of a sphere {label}"),
        ("m", 1, np.float32, "Msun", "Mass within a sphere {label}"),
        ("com", 3, np.float32, "Mpc", "Centre of mass within a sphere {label}"),
        (
            "vcom",
            3,
            np.float32,
            "km/s",
            "Centre of mass velocity within a sphere {label}",
        ),
        # gas properties
        ("Mgas", 1, np.float32, "Msun", "Total gas mass within a sphere {label}"),
        (
            "Jgas",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of gas within a sphere {label}",
        ),
        (
            "Mgasmetal",
            1,
            np.float32,
            "Msun",
            "Total metal mass of gas within a sphere {label}",
        ),
        (
            "Mhotgas",
            1,
            np.float32,
            "Msun",
            "Total mass of gas with T > 1e5 K within a sphere {label}",
        ),
        (
            "Tgas",
            1,
            np.float32,
            "K",
            "Mass-weighted average temperature of gas with T > 1e5 K within a sphere {label}",
        ),
        (
            "Xraylum",
            3,
            np.float64,
            "erg/s",
            "Total Xray luminosity within a sphere {label}",
        ),
        (
            "Xrayphlum",
            3,
            np.float64,
            "1/s",
            "Total Xray photon luminosity within a sphere {label}",
        ),
        ("compY", 1, np.float64, "cm**2", "Total Compton y within a sphere {label}"),
        # DM properties
        ("MDM", 1, np.float32, "Msun", "Total DM mass within a sphere {label}"),
        (
            "JDM",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of DM within a sphere {label}",
        ),
        # stellar properties
        ("Mstar", 1, np.float32, "Msun", "Total stellar mass within a sphere {label}"),
        (
            "Jstar",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of stars within a sphere {label}",
        ),
        (
            "Mstarinit",
            1,
            np.float32,
            "Msun",
            "Total initial stellar mass with a sphere {label}",
        ),
        (
            "Mstarmetal",
            1,
            np.float32,
            "Msun",
            "Total metal mass of stars within a sphere {label}",
        ),
        (
            "Lstar",
            9,
            np.float32,
            "dimensionless",
            "Total stellar luminosity within a sphere {label}",
        ),
        # BH properties
        (
            "MBHdyn",
            1,
            np.float32,
            "Msun",
            "Total dynamical BH mass within a sphere {label}",
        ),
        (
            "MBHsub",
            1,
            np.float32,
            "Msun",
            "Total sub-grid BH mass within a sphere {label}",
        ),
        (
            "BHlasteventa",
            1,
            np.float32,
            "dimensionless",
            "Last AGN feedback event within a sphere {label}",
        ),
        ("BHmaxM", 1, np.float32, "Msun", "Maximum BH mass within a sphere {label}"),
        (
            "BHmaxID",
            1,
            np.uint64,
            "dimensionless",
            "ID of most massive BH within a sphere {label}",
        ),
        (
            "BHmaxpos",
            3,
            np.float32,
            "Mpc",
            "Position of most massive BH within a sphere {label}",
        ),
        (
            "BHmaxvel",
            3,
            np.float32,
            "km/s",
            "Velocity of most massive BH within a sphere {label}",
        ),
        (
            "BHmaxAR",
            1,
            np.float32,
            "Msun/yr",
            "Accretion rate of most massive BH within a sphere {label}",
        ),
        (
            "BHmaxlasteventa",
            1,
            np.float32,
            "dimensionless",
            "Last AGN feedback event of the most massive BH within a sphere {label}",
        ),
    ]

    def __init__(self, cellgrid, SOval, type="mean"):
        super().__init__(cellgrid)

        if not type in ["mean", "crit", "physical", "BN98"]:
            raise AttributeError(f"Unknown SO type: {type}!")
        self.type = type

        # This specifies how large a sphere is read in:
        # we use default values that are sufficiently small/large to avoid reading in too many particles
        self.mean_density_multiple = 1000.0
        self.critical_density_multiple = 1000.0
        self.physical_radius_mpc = 0.0
        if type == "mean":
            self.mean_density_multiple = SOval
        elif type == "crit":
            self.critical_density_multiple = SOval
        elif type == "BN98":
            self.critical_density_multiple = cellgrid.virBN98
        elif type == "physical":
            self.physical_radius_mpc = 0.001 * SOval

        # Give this calculation a name so we can select it on the command line
        if type in ["mean", "crit"]:
            self.name = f"SO_{SOval:.0f}_{type}"
        elif type == "physical":
            self.name = f"SO_{SOval:.0f}_kpc"
        elif type == "BN98":
            self.name = "SO_BN98"

        # set some variables that are used during the calculation and that do not change
        if self.type == "crit":
            self.reference_density = (
                self.critical_density_multiple * self.critical_density
            )
            self.SO_name = f"{self.critical_density_multiple:.0f}_crit"
            self.label = f"within which the density is {self.critical_density_multiple:.0f} times the critical value"
        elif self.type == "mean":
            self.reference_density = self.mean_density_multiple * self.mean_density
            self.SO_name = f"{self.mean_density_multiple:.0f}_mean"
            self.label = f"within which the density is {self.mean_density_multiple:.0f} times the mean value"
        elif self.type == "physical":
            self.reference_density = 0.0
            self.SO_name = f"{1000. * self.physical_radius_mpc:.0f}_kpc"
            self.label = f"with a radius of {1000. * self.physical_radius_mpc:.0f} kpc"
        elif self.type == "BN98":
            self.reference_density = (
                self.critical_density_multiple * self.critical_density
            )
            self.SO_name = "BN98"
            self.label = f"within which the density is {self.critical_density_multiple:.2f} times the critical value"

    def calculate(self, input_halo, data, halo_result):
        """
        Compute spherical masses and overdensities for a halo

        input_halo       - dict with halo properties passed in from VR (see
                           halo_centres.py)
        data             - contains particle data. E.g. data["PartType1"]["Coordinates"]
                           has the particle coordinates for type 1
        halo_result      - dict with halo properties computed so far. Properties
                           computed here should be added to halo_result.

        Input particle data arrays are unyt_arrays.
        """

        # Find the halo centre of potential
        centre = input_halo["cofp"]

        # Make an array of particle masses, radii and positions
        mass = []
        radius = []
        position = []
        velocity = []
        types = []
        for ptype in data:
            mass.append(data[ptype][mass_dataset(ptype)])
            pos = data[ptype]["Coordinates"] - centre[None, :]
            position.append(pos)
            r = np.sqrt(np.sum(pos ** 2, axis=1))
            radius.append(r)
            velocity.append(data[ptype]["Velocities"])
            typearr = np.zeros(r.shape, dtype="U9")
            typearr[:] = ptype
            types.append(typearr)
        mass = unyt.array.uconcatenate(mass)
        radius = unyt.array.uconcatenate(radius)
        position = unyt.array.uconcatenate(position)
        velocity = unyt.array.uconcatenate(velocity)
        types = np.concatenate(types)

        # Sort by radius
        order = np.argsort(radius)
        ordered_radius = radius[order]
        cumulative_mass = np.cumsum(mass[order], dtype=np.float64).astype(mass.dtype)

        # Compute density within radius of each particle.
        # Will need to skip any at zero radius.
        nskip = 0
        while nskip < len(ordered_radius) and ordered_radius[nskip] == 0:
            nskip += 1
        ordered_radius = ordered_radius[nskip:]
        cumulative_mass = cumulative_mass[nskip:]
        nr_parts = len(ordered_radius)
        density = cumulative_mass / (4.0 / 3.0 * np.pi * ordered_radius ** 3)

        reg = mass.units.registry

        SO = {}
        # declare all the variables we will compute
        # we set them to 0 in case a particular variable cannot be computed
        # all variables are defined with physical units and an appropriate dtype
        # we need to use the custom unit registry so that everything can be converted
        # back to snapshot units in the end
        for name, shape, dtype, unit, _ in self.SO_properties:
            if shape > 1:
                val = [0] * shape
            else:
                val = 0
            SO[name] = unyt.unyt_array(val, dtype=dtype, units=unit, registry=reg)

        # Check if we ever reach the density threshold
        if self.reference_density > 0.0 * self.reference_density:
            if nr_parts > 0 and np.any(density > self.reference_density):
                # Find smallest radius where the density is below the threshold
                i = np.argmax(density <= self.reference_density)
                # Interpolate to get the actual radius
                if i == 0:
                    # we know that there are points above the threshold
                    # unfortunately, the centre is not
                    # find the furthest one that is:
                    i = np.nonzero(density > self.reference_density)[0][-1]
                    # use the next point as the "first one that is below the threshold"
                    i += 1
                r1 = ordered_radius[i - 1]
                r2 = ordered_radius[i]
                logrho1 = np.log10(density[i - 1].to(self.reference_density.units))
                logrho2 = np.log10(density[i].to(self.reference_density.units))
                # preserve the unyt_array dtype and units by using '+=' instead of assignment
                SO["r"] += r2 + (r2 - r1) * (
                    np.log10(self.reference_density) - logrho2
                ) / (logrho2 - logrho1)
                if SO["r"] > 2.0 * r2 or SO["r"] < 0.5 * r1:
                    raise RuntimeError(
                        f"Interpolation failed (r1: {r1.to('Mpc')}, r2: {r2.to('Mpc')}, rSO: {SO['r'].to('Mpc')})!"
                    )
                SO["m"] += 4.0 / 3.0 * np.pi * SO["r"] ** 3 * self.reference_density
        elif self.physical_radius_mpc > 0.0:
            SO["r"] += self.physical_radius_mpc * unyt.Mpc
            if nr_parts > 0:
                # find the enclosed mass using interpolation
                i = np.argmax(ordered_radius > SO["r"])
                if i == 0:
                    # we only have particles in the centre, so we cannot interpolate
                    SO["m"] += cumulative_mass[i]
                else:
                    r1 = ordered_radius[i - 1]
                    r2 = ordered_radius[i]
                    M1 = cumulative_mass[i - 1]
                    M2 = cumulative_mass[i]
                    SO["m"] += M1 + (SO["r"] - r1) / (r2 - r1) * (M2 - M1)

        else:
            # if we get here, we must be in the case where physical_radius_mpc is supposed to be 0
            # that can only happen if we are looking at a multiple of some radius
            # in that case, SO["r"] should remain 0
            # in any other case, something went wrong
            if not hasattr(self, "multiple"):
                raise ("Physical radius was set to 0! This should not happen!")

        if SO["r"] > 0.0 * radius.units:
            gas_selection = radius[types == "PartType0"] < SO["r"]
            dm_selection = radius[types == "PartType1"] < SO["r"]
            star_selection = radius[types == "PartType4"] < SO["r"]
            bh_selection = radius[types == "PartType5"] < SO["r"]

            all_selection = radius < SO["r"]
            mass = mass[all_selection]
            position = position[all_selection]
            velocity = velocity[all_selection]
            types = types[all_selection]

            # note that we cannot divide by mSO here, since that was based on an interpolation
            SO["com"][:] = (mass[:, None] * position).sum(axis=0) / mass.sum()
            SO["com"][:] += centre
            SO["vcom"][:] = (mass[:, None] * velocity).sum(axis=0) / mass.sum()

            gas_masses = mass[types == "PartType0"]
            gas_relpos = position[types == "PartType0"][:, :] - SO["com"][None, :]
            gas_relvel = velocity[types == "PartType0"][:, :] - SO["vcom"][None, :]
            SO["Mgas"] += gas_masses.sum()
            SO["Jgas"][:] = (
                gas_masses[:, None] * unyt.array.ucross(gas_relpos, gas_relvel)
            ).sum(axis=0)

            dm_masses = mass[types == "PartType1"]
            dm_relpos = position[types == "PartType1"][:, :] - SO["com"][None, :]
            dm_relvel = velocity[types == "PartType1"][:, :] - SO["vcom"][None, :]
            SO["MDM"] += dm_masses.sum()
            SO["JDM"][:] = (
                dm_masses[:, None] * unyt.array.ucross(dm_relpos, dm_relvel)
            ).sum(axis=0)

            star_masses = mass[types == "PartType4"]
            star_relpos = position[types == "PartType4"][:, :] - SO["com"][None, :]
            star_relvel = velocity[types == "PartType4"][:, :] - SO["vcom"][None, :]
            SO["Mstar"] += star_masses.sum()
            SO["Jstar"][:] = (
                star_masses[:, None] * unyt.array.ucross(star_relpos, star_relvel)
            ).sum(axis=0)

            SO["MBHdyn"] += mass[types == "PartType5"].sum()

            # gas specific properties. We (can) only do these if we have gas.
            # (remember that "PartType0" might not be part of 'data' at all)
            if np.any(gas_selection):
                SO["Mgasmetal"] += (
                    gas_masses * data["PartType0"]["MetalMassFractions"][gas_selection]
                ).sum()

                gas_temperatures = data["PartType0"]["Temperatures"][gas_selection]
                Tgas_selection = gas_temperatures > 1.0e5 * unyt.K
                SO["Mhotgas"] += gas_masses[Tgas_selection].sum()

                if np.any(Tgas_selection):
                    SO["Tgas"] += (
                        gas_temperatures[Tgas_selection] * gas_masses[Tgas_selection]
                    ).sum() / SO["Mhotgas"]

                SO["Xraylum"] += data["PartType0"]["XrayLuminosities"][
                    gas_selection
                ].sum()
                SO["Xrayphlum"] += data["PartType0"]["XrayPhotonLuminosities"][
                    gas_selection
                ].sum()

                SO["compY"] += data["PartType0"]["ComptonYParameters"][
                    gas_selection
                ].sum()

            # star specific properties
            if np.any(star_selection):
                SO["Mstarinit"] += data["PartType4"]["InitialMasses"][
                    star_selection
                ].sum()
                SO["Mstarmetal"] += (
                    star_masses
                    * data["PartType4"]["MetalMassFractions"][star_selection]
                ).sum()
                SO["Lstar"][:] = data["PartType4"]["Luminosities"][star_selection].sum()

            # BH specific properties
            if np.any(bh_selection):
                SO["MBHsub"] += data["PartType5"]["SubgridMasses"][bh_selection].sum()
                agn_eventa = data["PartType5"]["LastAGNFeedbackScaleFactors"][
                    bh_selection
                ]

                SO["BHlasteventa"] += np.max(agn_eventa)

                iBHmax = np.argmax(data["PartType5"]["SubgridMasses"][bh_selection])
                SO["BHmaxM"] += data["PartType5"]["SubgridMasses"][bh_selection][iBHmax]
                # unyt annoyingly converts to a floating point type if you use '+='
                # the only way to avoid this is by directly setting the data for the unyt_array
                SO["BHmaxID"].data = data["PartType5"]["ParticleIDs"][bh_selection][
                    iBHmax
                ].data
                SO["BHmaxpos"] += data["PartType5"]["Coordinates"][bh_selection][iBHmax]
                SO["BHmaxvel"] += data["PartType5"]["Velocities"][bh_selection][iBHmax]
                SO["BHmaxAR"] += data["PartType5"]["AccretionRates"][bh_selection][
                    iBHmax
                ]
                SO["BHmaxlasteventa"] += agn_eventa[iBHmax]

        # Return value should be a dict containing unyt_arrays and descriptions.
        # The dict keys will be used as HDF5 dataset names in the output.
        for name, _, _, _, description in self.SO_properties:
            halo_result.update(
                {
                    f"SO/{self.SO_name}/{name}": (
                        SO[name],
                        description.format(label=self.label),
                    )
                }
            )

        return


class RadiusMultipleSOProperties(SOProperties):
    def __init__(self, cellgrid, SOval, multiple, type="mean"):
        if not type in ["mean", "crit"]:
            raise AttributeError(
                "SOs with a radius that is a multiple of another SO radius are only allowed for type mean or crit!"
            )

        # initialise the SOProperties object using a conservative physical radius estimate
        super().__init__(cellgrid, 3000.0, "physical")

        # overwrite the name, SO_name and label
        self.SO_name = f"{multiple:.0f}xR_{SOval:.0f}_{type}"
        self.label = f"with a radius that is {self.SO_name}"
        self.name = f"SO_{self.SO_name}"

        self.requested_type = type
        self.requested_SOval = SOval
        self.multiple = multiple

    def calculate(self, input_halo, data, halo_result):

        # find the actual physical radius we want
        key = f"SO/{self.requested_SOval:.0f}_{self.requested_type}/r"
        if not key in halo_result:
            raise RuntimeError(
                f"Trying to obtain {key}, but the corresponding SO radius has not been calculated!"
            )
        self.physical_radius_mpc = self.multiple * (halo_result[key][0].to("Mpc").value)

        # Check that we read in a large enough radius
        if self.multiple*halo_result[key][0] > input_halo["read_radius"]:
            raise ReadRadiusTooSmallException("SO radius multiple estimate was too small!")

        super().calculate(input_halo, data, halo_result)
        return
