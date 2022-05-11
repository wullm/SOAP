#!/bin/env python

import numpy as np
import unyt
from halo_properties import HaloProperty

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

    def __init__(self, cellgrid, SOval, type="mean"):
        super().__init__(cellgrid)

        if not type in ["mean", "crit", "physical"]:
            raise AttributeError(f"Unknown SO type: {type}!")
        self.type = type

        # This specifies how large a sphere is read in:
        if type in ["mean", "crit"]:
            self.mean_density_multiple = SOval
            self.critical_density_multiple = SOval
            self.physical_radius_mpc = 0.0
        elif type == "physical":
            # use a large multiple to avoid reading in too many particles
            self.mean_density_multiple = 1000.0
            self.critical_density_multiple = 1000.0
            self.physical_radius_mpc = 0.001 * SOval

        # Give this calculation a name so we can select it on the command line
        self.name = "SO_%d_%s" % (SOval, type)

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

        if self.type == "crit":
            reference_density = self.critical_density_multiple * self.critical_density
            name = f"{self.critical_density_multiple:.0f}_crit"
            label = f"within which the density is {self.critical_density_multiple:.0f} times the critical value"
        elif self.type == "mean":
            reference_density = self.mean_density_multiple * self.mean_density
            name = f"{self.mean_density_multiple:.0f}_mean"
            label = f"within which the density is {self.mean_density_multiple:.0f} times the mean value"
        elif self.type == "physical":
            reference_density = 0.0
            name = f"{1000. * self.physical_radius_mpc:.0f}_kpc"
            label = f"with a radius of {1000. * self.physical_radius_mpc:.0f} kpc"

        reg = mass.units.registry

        # declare all the variables we will compute
        # we set them to 0 in case a particular variable cannot be computed
        # all variables are defined with physical units and an appropriate dtype
        # we need to use the custom unit registry so that everything can be converted
        # back to snapshot units in the end
        rSO = unyt.unyt_array(0.0, dtype=np.float32, units="Mpc", registry=reg)
        mSO = unyt.unyt_array(0.0, dtype=np.float32, units="Msun", registry=reg)
        comSO = unyt.unyt_array(
            [0.0, 0.0, 0.0], dtype=np.float32, units="Mpc", registry=reg
        )
        vcomSO = unyt.unyt_array(
            [0.0, 0.0, 0.0], dtype=np.float32, units="km/s", registry=reg
        )

        MgasSO = unyt.unyt_array(0.0, dtype=np.float32, units="Msun", registry=reg)
        Jgas = unyt.unyt_array(
            [0.0, 0.0, 0.0], dtype=np.float32, units="Msun*kpc*km/s", registry=reg
        )
        MhotgasSO = unyt.unyt_array(0.0, dtype=np.float32, units="Msun", registry=reg)
        TgasSO = unyt.unyt_array(0.0, dtype=np.float32, units="K", registry=reg)
        Mgasmetal = unyt.unyt_array(0.0, dtype=np.float32, units="Msun", registry=reg)
        XraylumSO = unyt.unyt_array(0.0, dtype=np.float64, units="erg/s", registry=reg)
        XrayphlumSO = unyt.unyt_array(0.0, dtype=np.float64, units="1/s", registry=reg)
        compYSO = unyt.unyt_array(0.0, dtype=np.float64, units="cm**2", registry=reg)

        MdmSO = unyt.unyt_array(0.0, dtype=np.float32, units="Msun", registry=reg)
        JDM = unyt.unyt_array(
            [0.0, 0.0, 0.0], dtype=np.float32, units="Msun*kpc*km/s", registry=reg
        )

        MstarSO = unyt.unyt_array(0.0, dtype=np.float32, units="Msun", registry=reg)
        Jstar = unyt.unyt_array(
            [0.0, 0.0, 0.0], dtype=np.float32, units="Msun*kpc*km/s", registry=reg
        )
        MstarinitSO = unyt.unyt_array(0.0, dtype=np.float32, units="Msun", registry=reg)
        Mstarmetal = unyt.unyt_array(0.0, dtype=np.float32, units="Msun", registry=reg)
        Lstar = unyt.unyt_array(
            [0.0] * 9, dtype=np.float32, units="dimensionless", registry=reg
        )

        MBHdynSO = unyt.unyt_array(0.0, dtype=np.float32, units="Msun", registry=reg)
        MBHsubSO = unyt.unyt_array(0.0, dtype=np.float32, units="Msun", registry=reg)
        BHlasteventa = unyt.unyt_array(
            0.0, dtype=np.float32, units="dimensionless", registry=reg
        )
        BHmaxM = unyt.unyt_array(0.0, dtype=np.float32, units="Msun", registry=reg)
        BHmaxID = unyt.unyt_array(
            0.0, dtype=np.uint64, units="dimensionless", registry=reg
        )
        BHmaxpos = unyt.unyt_array(
            [0.0, 0.0, 0.0], dtype=np.float32, units="Mpc", registry=reg
        )
        BHmaxvel = unyt.unyt_array(
            [0.0, 0.0, 0.0], dtype=np.float32, units="km/s", registry=reg
        )
        BHmaxAR = unyt.unyt_array(0.0, dtype=np.float32, units="Msun/yr", registry=reg)
        BHmaxlasteventa = unyt.unyt_array(
            0.0, dtype=np.float32, units="dimensionless", registry=reg
        )

        # Check if we ever reach the density threshold
        if reference_density > 0.0 * reference_density:
            if nr_parts > 0 and np.any(density > reference_density):
                # Find smallest radius where the density is below the threshold
                i = np.argmax(density <= reference_density)
                # Interpolate to get the actual radius
                if i == 0:
                    raise RuntimeError("This should not happen!")
                r1 = ordered_radius[i - 1]
                r2 = ordered_radius[i]
                logrho1 = np.log10(density[i - 1].to(reference_density.units))
                logrho2 = np.log10(density[i].to(reference_density.units))
                # preserve the unyt_array dtype and units by using '+=' instead of assignment
                rSO += r2 + (r2 - r1) * (np.log10(reference_density) - logrho2) / (
                    logrho2 - logrho1
                )
                if rSO > r2 or rSO < r1:
                    raise RuntimeError(f"Interpolation failed!")
                mSO += 4.0 / 3.0 * np.pi * rSO ** 3 * reference_density
        elif self.physical_radius_mpc > 0.0:
            rSO += self.physical_radius_mpc * unyt.Mpc
            if nr_parts > 0:
                # find the enclosed mass using interpolation
                i = np.argmax(ordered_radius > rSO)
                if i == 0:
                    # we only have particles in the centre, so we cannot interpolate
                    mSO += cumulative_mass[i]
                else:
                    r1 = ordered_radius[i - 1]
                    r2 = ordered_radius[i]
                    M1 = cumulative_mass[i - 1]
                    M2 = cumulative_mass[i]
                    mSO += M1 + (rSO - r1) / (r2 - r1) * (M2 - M1)
        else:
            raise RuntimeError("Should not happen!")

        if rSO > 0.0 * radius.units:
            gas_selection = radius[types == "PartType0"] < rSO
            dm_selection = radius[types == "PartType1"] < rSO
            star_selection = radius[types == "PartType4"] < rSO
            bh_selection = radius[types == "PartType5"] < rSO

            all_selection = radius < rSO
            mass = mass[all_selection]
            position = position[all_selection]
            velocity = velocity[all_selection]
            types = types[all_selection]

            # note that we cannot divide by mSO here, since that was based on an interpolation
            comSO[:] = (mass[:, None] * position).sum(axis=0) / mass.sum()
            comSO[:] += centre
            vcomSO[:] = (mass[:, None] * velocity).sum(axis=0) / mass.sum()

            gas_masses = mass[types == "PartType0"]
            gas_relpos = position[types == "PartType0"][:, :] - comSO[None, :]
            gas_relvel = velocity[types == "PartType0"][:, :] - vcomSO[None, :]
            MgasSO += gas_masses.sum()
            Jgas[:] = (
                gas_masses[:, None] * unyt.array.ucross(gas_relpos, gas_relvel)
            ).sum(axis=0)

            dm_masses = mass[types == "PartType1"]
            dm_relpos = position[types == "PartType1"][:, :] - comSO[None, :]
            dm_relvel = velocity[types == "PartType1"][:, :] - vcomSO[None, :]
            MdmSO += dm_masses.sum()
            JDM[:] = (dm_masses[:, None] * unyt.array.ucross(dm_relpos, dm_relvel)).sum(
                axis=0
            )

            star_masses = mass[types == "PartType4"]
            star_relpos = position[types == "PartType4"][:, :] - comSO[None, :]
            star_relvel = velocity[types == "PartType4"][:, :] - vcomSO[None, :]
            MstarSO += star_masses.sum()
            Jstar[:] = (
                star_masses[:, None] * unyt.array.ucross(star_relpos, star_relvel)
            ).sum(axis=0)

            MBHdynSO += mass[types == "PartType5"].sum()

            # gas specific properties. We (can) only do these if we have gas.
            # (remember that "PartType0" might not be part of 'data' at all)
            if np.any(gas_selection):
                gas_temperatures = data["PartType0"]["Temperatures"][gas_selection]
                Tgas_selection = gas_temperatures > 1.0e5 * unyt.K
                MhotgasSO += gas_masses[Tgas_selection].sum()
                Mgasmetal += (
                    gas_masses * data["PartType0"]["MetalMassFractions"][gas_selection]
                ).sum()

                if np.any(Tgas_selection):
                    TgasSO += (
                        gas_temperatures[Tgas_selection] * gas_masses[Tgas_selection]
                    ).sum() / MhotgasSO

                XraylumSO += data["PartType0"]["XrayLuminosities"][gas_selection].sum()
                XrayphlumSO += data["PartType0"]["XrayPhotonLuminosities"][
                    gas_selection
                ].sum()

                compYSO += data["PartType0"]["ComptonYParameters"][gas_selection].sum()

            # star specific properties
            if np.any(star_selection):
                MstarinitSO += data["PartType4"]["InitialMasses"][star_selection].sum()
                Mstarmetal += (
                    star_masses
                    * data["PartType4"]["MetalMassFractions"][star_selection]
                ).sum()
                Lstar[:] = data["PartType4"]["Luminosities"][star_selection].sum()

            # BH specific properties
            if np.any(bh_selection):
                MBHsubSO += data["PartType5"]["SubgridMasses"][bh_selection].sum()
                agn_eventa = data["PartType5"]["LastAGNFeedbackScaleFactors"][
                    bh_selection
                ]

                BHlasteventa += np.max(agn_eventa)

                iBHmax = np.argmax(data["PartType5"]["SubgridMasses"][bh_selection])
                BHmaxM += data["PartType5"]["SubgridMasses"][bh_selection][iBHmax]
                BHmaxID += data["PartType5"]["ParticleIDs"][bh_selection][iBHmax]
                BHmaxpos += data["PartType5"]["Coordinates"][bh_selection][iBHmax]
                BHmaxvel += data["PartType5"]["Velocities"][bh_selection][iBHmax]
                BHmaxAR += data["PartType5"]["AccretionRates"][bh_selection][iBHmax]
                BHmaxlasteventa += agn_eventa[iBHmax]

        # Return value should be a dict containing unyt_arrays and descriptions.
        # The dict keys will be used as HDF5 dataset names in the output.
        halo_result.update(
            {
                f"SO/{name}/r": (rSO, f"Radius {label}"),
                f"SO/{name}/m": (mSO, f"Mass within a sphere {label}"),
                f"SO/{name}/com": (comSO, f"Centre of mass within a sphere {label}"),
                f"SO/{name}/vcom": (
                    vcomSO,
                    f"Centre of mass velocity within a sphere {label}",
                ),
                f"SO/{name}/Mgas": (MgasSO, f"Total gas mass within a sphere {label}"),
                f"SO/{name}/Jgas": (
                    Jgas,
                    f"Total angular momentum of gas within a sphere {label}",
                ),
                f"SO/{name}/Mhotgas": (
                    MhotgasSO,
                    f"Total mass of gas with T > 1e5 K within a sphere {label}",
                ),
                f"SO/{name}/Tgas": (
                    TgasSO,
                    f"Mass-weighted average temperature of gas with T > 1e5 K within a sphere {label}",
                ),
                f"SO/{name}/Mgasmetal": (
                    Mgasmetal,
                    f"Total metal mass of gas within a sphere {label}",
                ),
                f"SO/{name}/Xraylum": (
                    XraylumSO,
                    f"Total Xray luminosity within a sphere {label}",
                ),
                f"SO/{name}/Xrayphlum": (
                    XrayphlumSO,
                    f"Total Xray photon luminosity within a sphere {label}",
                ),
                f"SO/{name}/compY": (
                    compYSO,
                    f"Total Compton y within a sphere {label}",
                ),
                f"SO/{name}/Mdm": (MdmSO, f"Total DM mass within a sphere {label}"),
                f"SO/{name}/JDM": (
                    JDM,
                    f"Total angular momentum of DM within a sphere {label}",
                ),
                f"SO/{name}/Mstar": (
                    MstarSO,
                    f"Total stellar mass within a sphere {label}",
                ),
                f"SO/{name}/Jstar": (
                    Jstar,
                    f"Total angular momentum of stars within a sphere {label}",
                ),
                f"SO/{name}/Mstarinit": (
                    MstarinitSO,
                    f"Total initial stellar mass with a sphere {label}",
                ),
                f"SO/{name}/Mstarmetal": (
                    Mstarmetal,
                    f"Total metal mass of stars within a sphere {label}",
                ),
                f"SO/{name}/Lstar": (
                    Lstar,
                    f"Total stellar luminosity within a sphere {label}",
                ),
                f"SO/{name}/MBHdyn": (
                    MBHdynSO,
                    f"Total dynamical BH mass within a sphere {label}",
                ),
                f"SO/{name}/MBHsub": (
                    MBHsubSO,
                    f"Total sub-grid BH mass within a sphere {label}",
                ),
                f"SO/{name}/BHlasteventa": (
                    BHlasteventa,
                    f"Last AGN feedback event within a sphere {label}",
                ),
                f"SO/{name}/BHmaxM": (
                    BHmaxM,
                    f"Maximum BH mass within a sphere {label}",
                ),
                f"SO/{name}/BHmaxID": (
                    BHmaxID,
                    f"ID of most massive BH within a sphere {label}",
                ),
                f"SO/{name}/BHmaxpos": (
                    BHmaxpos,
                    f"Position of most massive BH within a sphere {label}",
                ),
                f"SO/{name}/BHmaxvel": (
                    BHmaxvel,
                    f"Velocity of most massive BH within a sphere {label}",
                ),
                f"SO/{name}/BHmaxAR": (
                    BHmaxAR,
                    f"Accretion rate of most massive BH within a sphere {label}",
                ),
                f"SO/{name}/BHmaxlasteventa": (
                    BHmaxlasteventa,
                    f"Last AGN feedback event of the most massive BH within a sphere {label}",
                ),
            }
        )
