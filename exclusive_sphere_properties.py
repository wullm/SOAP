#!/bin/env python

import numpy as np
import unyt

from halo_properties import HaloProperty
from dataset_names import mass_dataset
from half_mass_radius import get_half_mass_radius

from astropy.cosmology import w0waCDM, z_at_value
import astropy.constants as const
import astropy.units as astropy_units

# index of elements O and Fe in the SmoothedElementMassFractions dataset
indexO = 4
indexFe = 8


class RecentlyHeatedGasFilter:
    """
    Filter used to determine whether gas particles should be considered to be
    "recently heated".

    This corresponds to the lightcone map filter used in SWIFT itself, which
    filters out gas particles for which LastAGNFeedbackScaleFactors is less
    than 15 Myr ago, and within some temperature bracket.

    Since the conversion from a time difference to a scale factor is not
    trivial, we compute the corresponding scale factor limit only once using
    the correct astropy.cosmology.
    """

    def __init__(
        self,
        cellgrid,
        delta_time=15.0 * unyt.Myr,
        delta_logT_min=-1.0,
        delta_logT_max=0.3,
        AGN_delta_T=8.80144197177e7 * unyt.K,
    ):
        H0 = unyt.unyt_quantity(
            cellgrid.cosmology["H0 [internal units]"],
            units="1/snap_time",
            registry=cellgrid.snap_unit_registry,
        ).to("1/s")

        Omega_b = cellgrid.cosmology["Omega_b"]
        Omega_lambda = cellgrid.cosmology["Omega_lambda"]
        Omega_r = cellgrid.cosmology["Omega_r"]
        Omega_m = cellgrid.cosmology["Omega_m"]
        w_0 = cellgrid.cosmology["w_0"]
        w_a = cellgrid.cosmology["w_a"]
        z_now = cellgrid.cosmology["Redshift"]

        # expressions taken directly from astropy, since they do no longer
        # allow access to these attributes (since version 5.1+)
        critdens_const = (3.0 / (8.0 * np.pi * const.G)).cgs.value
        a_B_c2 = (4.0 * const.sigma_sb / const.c**3).cgs.value

        # SWIFT provides Omega_r, but we need a consistent Tcmb0 for astropy.
        # This is an exact inversion of the procedure performed in astropy.
        critical_density_0 = astropy_units.Quantity(
            critdens_const * H0.to("1/s").value ** 2,
            astropy_units.g / astropy_units.cm**3,
        )

        Tcmb0 = (Omega_r * critical_density_0.value / a_B_c2) ** (1.0 / 4.0)

        cosmology = w0waCDM(
            H0=H0.to_astropy(),
            Om0=Omega_m,
            Ode0=Omega_lambda,
            w0=w_0,
            wa=w_a,
            Tcmb0=Tcmb0,
            Ob0=Omega_b,
        )

        lookback_time_now = cosmology.lookback_time(z_now)
        lookback_time_limit = lookback_time_now + delta_time.to_astropy()
        z_limit = z_at_value(cosmology.lookback_time, lookback_time_limit)

        # for some reason, the return type of z_at_value has changed between
        # astropy versions. We make sure it is not some astropy quantity
        # before using it.
        if hasattr(z_limit, "value"):
            z_limit = z_limit.value

        self.a_limit = 1.0 / (1.0 + z_limit) * unyt.dimensionless

        self.Tmin = AGN_delta_T * 10.0**delta_logT_min
        self.Tmax = AGN_delta_T * 10.0**delta_logT_max

    def is_recently_heated(self, lastAGNfeedback, temperature):
        return (
            (lastAGNfeedback >= self.a_limit)
            & (temperature >= self.Tmin)
            & (temperature <= self.Tmax)
        )


class ExclusiveSphereProperties(HaloProperty):

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

    def __init__(self, cellgrid, physical_radius_kpc, recently_heated_gas_filter):
        super().__init__(cellgrid)

        self.filter = recently_heated_gas_filter

        # This specifies how large a sphere is read in:
        self.mean_density_multiple = None
        self.critical_density_multiple = None

        # Minimum physical radius to read in (pMpc)
        self.physical_radius_mpc = 0.001 * physical_radius_kpc

        self.name = f"exclusive_sphere_{physical_radius_kpc:.0f}kpc"

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

        gas_mask_ap = mask[types == "PartType0"]
        dm_mask_ap = mask[types == "PartType1"]
        star_mask_ap = mask[types == "PartType4"]
        bh_mask_ap = mask[types == "PartType5"]

        Ngas = unyt.unyt_array(
            gas_mask_ap.sum(),
            dtype=np.uint32,
            units="dimensionless",
            registry=mass.units.registry,
        )
        Ndm = unyt.unyt_array(
            dm_mask_ap.sum(),
            dtype=np.uint32,
            units="dimensionless",
            registry=mass.units.registry,
        )
        Nstar = unyt.unyt_array(
            star_mask_ap.sum(),
            dtype=np.uint32,
            units="dimensionless",
            registry=mass.units.registry,
        )
        Nbh = unyt.unyt_array(
            bh_mask_ap.sum(),
            dtype=np.uint32,
            units="dimensionless",
            registry=mass.units.registry,
        )

        mass_gas = mass[type == "PartType0"]
        mass_dm = mass[type == "PartType1"]
        mass_star = mass[type == "PartType4"]

        pos_gas = position[type == "PartType0"]
        pos_dm = position[type == "PartType1"]
        pos_star = position[type == "PartType4"]

        vel_gas = velocity[type == "PartType0"]
        vel_dm = velocity[type == "PartType1"]
        vel_star = velocity[type == "PartType4"]

        Mtot = mass.sum()
        Mgas = mass_gas.sum()
        Mdm = mass_dm.sum()
        Mstar = mass_star.sum()
        if Nstar > 0:
            star_mask_all = data["PartType4"]["GroupNr_bound"] == index
            Mstar_init = data["PartType4"]["InitialMasses"][star_mask_all][
                star_mask_ap
            ].sum()
            lum = data["PartType4"]["Luminosities"][star_mask_all][star_mask_ap].sum(
                axis=0
            )
            Mstarmetal = (
                mass_star
                * data["PartType4"]["MetalMassFractions"][star_mask_all][star_mask_ap]
            ).sum()
        else:
            Mstar_init = unyt.unyt_array(Mstar, dtype=Mstar.dtype, units=Mstar.units)
            lum = unyt.unyt_array(
                [0.0] * 9,
                dtype=np.float32,
                units="dimensionless",
                registry=mass.units.registry,
            )
            Mstarmetal = unyt.unyt_array(Mstar, dtype=Mstar.dtype, units=Mstar.units)
        Mbh = mass[type == "PartType5"].sum()
        if Nbh > 0:
            bh_mask_all = data["PartType5"]["GroupNr_bound"] == index
            Mbh_subgrid = data["PartType5"]["SubgridMasses"][bh_mask_all][
                bh_mask_ap
            ].sum()

            agn_eventa = data["PartType5"]["LastAGNFeedbackScaleFactors"][bh_mask_all][
                bh_mask_ap
            ]

            BHlasteventa = np.max(agn_eventa)

            iBHmax = np.argmax(
                data["PartType5"]["SubgridMasses"][bh_mask_all][bh_mask_ap]
            )
            BHmaxM = data["PartType5"]["SubgridMasses"][bh_mask_all][bh_mask_ap][iBHmax]
            BHmaxID = data["PartType5"]["ParticleIDs"][bh_mask_all][bh_mask_ap][
                iBHmax
            ].value
            BHmaxpos = data["PartType5"]["Coordinates"][bh_mask_all][bh_mask_ap][iBHmax]
            BHmaxvel = data["PartType5"]["Velocities"][bh_mask_all][bh_mask_ap][iBHmax]
            BHmaxAR = data["PartType5"]["AccretionRates"][bh_mask_all][bh_mask_ap][
                iBHmax
            ]
            BHmaxlasteventa = agn_eventa[iBHmax]
        else:
            Mbh_subgrid = unyt.unyt_array(Mbh, dtype=Mbh.dtype, units=Mbh.units)
            BHlasteventa = 0.0
            BHmaxM = 0.0
            BHmaxID = 0
            BHmaxpos = [0.0, 0.0, 0.0]
            BHmaxvel = [0.0, 0.0, 0.0]
            BHmaxAR = 0.0
            BHmaxlasteventa = 0.0
        BHlasteventa = unyt.unyt_array(
            BHlasteventa,
            dtype=np.float32,
            units="dimensionless",
            registry=mass.units.registry,
        )
        BHmaxM = unyt.unyt_array(
            BHmaxM, dtype=np.float32, units="Msun", registry=mass.units.registry
        )
        BHmaxID = unyt.unyt_array(
            BHmaxID,
            dtype=np.uint64,
            units="dimensionless",
            registry=mass.units.registry,
        )
        BHmaxpos = unyt.unyt_array(
            BHmaxpos, dtype=np.float64, units="kpc", registry=mass.units.registry
        )
        BHmaxvel = unyt.unyt_array(
            BHmaxvel, dtype=np.float32, units="km/s", registry=mass.units.registry
        )
        BHmaxAR = unyt.unyt_array(
            BHmaxAR, dtype=np.float32, units="Msun/yr", registry=mass.units.registry
        )
        BHmaxlasteventa = unyt.unyt_array(
            BHmaxlasteventa,
            dtype=np.float32,
            units="dimensionless",
            registry=mass.units.registry,
        )

        com = unyt.unyt_array(
            [0.0] * 3, dtype=np.float32, units="Mpc", registry=mass.units.registry
        )
        vcom = unyt.unyt_array(
            [0.0] * 3, dtype=np.float32, units="km/s", registry=mass.units.registry
        )
        if Mtot > 0.0 * Mtot.units:
            com[:] = (mass[:, None] * position).sum(axis=0) / Mtot
            com[:] += centre
            vcom[:] = ((mass[:, None] / Mtot) * velocity).sum(axis=0)

        gas_kappa_corot = unyt.unyt_array(
            0.0, dtype=np.float32, units="dimensionless", registry=mass.units.registry
        )
        totLgas = unyt.unyt_array(
            [0.0] * 3,
            dtype=np.float64,
            units="Msun*kpc*km/s",
            registry=mass.units.registry,
        )
        veldisp_gas = unyt.unyt_array(
            [0.0] * 6,
            dtype=np.float32,
            units="km**2/s**2",
            registry=mass.units.registry,
        )
        if Mgas > 0.0 * Mgas.units:
            frac_mgas = mass_gas / Mgas
            com_gas = (frac_mgas[:, None] * pos_gas).sum(axis=0)
            vcom_gas = (frac_mgas[:, None] * vel_gas).sum(axis=0)
            gas_relpos = pos_gas - com_gas[None, :]
            gas_relvel = vel_gas - vcom_gas[None, :]
            Lgas = mass_gas[:, None] * unyt.array.ucross(gas_relpos, gas_relvel)
            totLgas[:] = Lgas.sum(axis=0)
            Lnrm = unyt.array.unorm(totLgas)
            if Lnrm > 0.0 * Lnrm.units:
                K = gas_relvel.astype(np.float64)
                K = 0.5 * (mass_gas[:, None] * K**2).sum()
                if K > 0.0 * K.units:
                    Li = ((Lgas / Lnrm) * totLgas[None, :]).sum(axis=1)
                    gas_r2 = (
                        gas_relpos[:, 0] ** 2
                        + gas_relpos[:, 1] ** 2
                        + gas_relpos[:, 2] ** 2
                    )
                    rdotL = (gas_relpos * totLgas[None, :]).sum(axis=1) / Lnrm
                    Ri2 = gas_r2 - rdotL**2
                    Krot = 0.5 * (Li**2 / (mass_gas * Ri2))
                    Kcorot = Krot[Li > 0.0].sum()
                    gas_kappa_corot += Kcorot / K

            vrel = vel_gas - vcom[None, :]
            veldisp_gas[0] += (frac_mgas * vrel[:, 0] * vrel[:, 0]).sum()
            veldisp_gas[1] += (frac_mgas * vrel[:, 1] * vrel[:, 1]).sum()
            veldisp_gas[2] += (frac_mgas * vrel[:, 2] * vrel[:, 2]).sum()
            veldisp_gas[3] += (frac_mgas * vrel[:, 0] * vrel[:, 1]).sum()
            veldisp_gas[4] += (frac_mgas * vrel[:, 0] * vrel[:, 2]).sum()
            veldisp_gas[5] += (frac_mgas * vrel[:, 1] * vrel[:, 2]).sum()

        totLdm = unyt.unyt_array(
            [0.0] * 3,
            dtype=np.float64,
            units="Msun*kpc*km/s",
            registry=mass.units.registry,
        )
        veldisp_dm = unyt.unyt_array(
            [0.0] * 6,
            dtype=np.float32,
            units="km**2/s**2",
            registry=mass.units.registry,
        )
        if Mdm > 0.0 * Mdm.units:
            frac_mdm = mass_dm / Mdm
            com_dm = (frac_mdm[:, None] * pos_dm).sum(axis=0)
            vcom_dm = (frac_mdm[:, None] * vel_dm).sum(axis=0)
            dm_relpos = pos_dm - com_dm[None, :]
            dm_relvel = vel_dm - vcom_dm[None, :]
            Ldm = mass_dm[:, None] * unyt.array.ucross(dm_relpos, dm_relvel)
            totLdm[:] = Ldm.sum(axis=0)

            vrel = vel_dm - vcom[None, :]
            veldisp_dm[0] += (frac_mdm * vrel[:, 0] * vrel[:, 0]).sum()
            veldisp_dm[1] += (frac_mdm * vrel[:, 1] * vrel[:, 1]).sum()
            veldisp_dm[2] += (frac_mdm * vrel[:, 2] * vrel[:, 2]).sum()
            veldisp_dm[3] += (frac_mdm * vrel[:, 0] * vrel[:, 1]).sum()
            veldisp_dm[4] += (frac_mdm * vrel[:, 0] * vrel[:, 2]).sum()
            veldisp_dm[5] += (frac_mdm * vrel[:, 1] * vrel[:, 2]).sum()

        star_kappa_corot = unyt.unyt_array(
            0.0, dtype=np.float32, units="dimensionless", registry=mass.units.registry
        )
        totLstar = unyt.unyt_array(
            [0.0] * 3,
            dtype=np.float64,
            units="Msun*kpc*km/s",
            registry=mass.units.registry,
        )
        veldisp_star = unyt.unyt_array(
            [0.0] * 6,
            dtype=np.float32,
            units="km**2/s**2",
            registry=mass.units.registry,
        )
        if Mstar > 0.0 * Mstar.units:
            frac_mstar = mass_star / Mstar
            com_star = (frac_mstar[:, None] * pos_star).sum(axis=0)
            vcom_star = (frac_mstar[:, None] * vel_star).sum(axis=0)
            star_relpos = pos_star - com_star[None, :]
            star_relvel = vel_star - vcom_star[None, :]
            Lstar = mass_star[:, None] * unyt.array.ucross(star_relpos, star_relvel)
            totLstar[:] = Lstar.sum(axis=0)
            Lnrm = unyt.array.unorm(totLstar)
            if Lnrm > 0.0 * Lnrm.units:
                K = star_relvel.astype(np.float64)
                K = 0.5 * (mass_star[:, None] * K**2).sum()
                if K > 0.0 * K.units:
                    Li = ((Lstar / Lnrm) * totLstar[None, :]).sum(axis=1)
                    star_r2 = (
                        star_relpos[:, 0] ** 2
                        + star_relpos[:, 1] ** 2
                        + star_relpos[:, 2] ** 2
                    )
                    rdotL = (star_relpos * totLstar[None, :]).sum(axis=1) / Lnrm
                    Ri2 = star_r2 - rdotL**2
                    Krot = 0.5 * (Li**2 / (mass_star * Ri2))
                    Kcorot = Krot[Li > 0.0].sum()
                    star_kappa_corot += Kcorot / K

            vrel = vel_star - vcom[None, :]
            veldisp_star[0] += (frac_mstar * vrel[:, 0] * vrel[:, 0]).sum()
            veldisp_star[1] += (frac_mstar * vrel[:, 1] * vrel[:, 1]).sum()
            veldisp_star[2] += (frac_mstar * vrel[:, 2] * vrel[:, 2]).sum()
            veldisp_star[3] += (frac_mstar * vrel[:, 0] * vrel[:, 1]).sum()
            veldisp_star[4] += (frac_mstar * vrel[:, 0] * vrel[:, 2]).sum()
            veldisp_star[5] += (frac_mstar * vrel[:, 1] * vrel[:, 2]).sum()

        SFR = 0.0
        Tgas = 0.0
        Tgas_no_agn = 0.0
        if Ngas > 0:
            gas_mask_all = data["PartType0"]["GroupNr_bound"] == index
            SFR = data["PartType0"]["StarFormationRates"][gas_mask_all][gas_mask_ap]
            # negative values of SFR are not SFR at all!
            is_SFR = SFR > 0.0
            SFR = SFR[is_SFR].sum()
            Mgas_SFR = mass_gas[is_SFR].sum()
            Mgas_noSFR = mass_gas[~is_SFR].sum()
            Mgasmetal = (
                mass_gas
                * data["PartType0"]["MetalMassFractions"][gas_mask_all][gas_mask_ap]
            )
            Mgasmetal_SFR = Mgasmetal[is_SFR].sum()
            Mgasmetal_noSFR = Mgasmetal[~is_SFR].sum()
            Mgasmetal = Mgasmetal.sum()
            MgasO = (
                mass_gas
                * data["PartType0"]["SmoothedElementMassFractions"][gas_mask_all][
                    gas_mask_ap
                ][:, indexO]
            )
            MgasO_SFR = MgasO[is_SFR].sum()
            MgasO_noSFR = MgasO[~is_SFR].sum()
            MgasO = MgasO.sum()
            MgasFe = (
                mass_gas
                * data["PartType0"]["SmoothedElementMassFractions"][gas_mask_all][
                    gas_mask_ap
                ][:, indexFe]
            )
            MgasFe_SFR = MgasFe[is_SFR].sum()
            MgasFe_noSFR = MgasFe[~is_SFR].sum()
            MgasFe = MgasFe.sum()
            gas_temp = data["PartType0"]["Temperatures"][gas_mask_all][gas_mask_ap]
            last_agn_gas = data["PartType0"]["LastAGNFeedbackScaleFactors"][
                gas_mask_all
            ][gas_mask_ap]
            no_agn = ~self.filter.is_recently_heated(last_agn_gas, gas_temp)
            Tgas = ((mass_gas / Mgas) * gas_temp).sum()
            if np.any(no_agn):
                mass_gas_no_agn = mass_gas[no_agn]
                Mgas_no_agn = mass_gas_no_agn.sum()
                if Mgas_no_agn > 0.0:
                    Tgas_no_agn = (mass_gas_no_agn / Mgas_no_agn) * gas_temp[no_agn]
        else:
            Mgasmetal_SFR = unyt.unyt_array(Mgas)
            Mgasmetal_noSFR = unyt.unyt_array(Mgas)
            Mgasmetal = unyt.unyt_array(Mgas)
            MgasO_SFR = unyt.unyt_array(Mgas)
            MgasO_noSFR = unyt.unyt_array(Mgas)
            MgasO = unyt.unyt_array(Mgas)
            MgasFe_SFR = unyt.unyt_array(Mgas)
            MgasFe_noSFR = unyt.unyt_array(Mgas)
            MgasFe = unyt.unyt_array(Mgas)
        SFR = unyt.unyt_array(
            SFR, dtype=np.float32, units="Msun/yr", registry=mass.units.registry
        )
        Tgas = unyt.unyt_array(
            Tgas, dtype=np.float32, units="K", registry=mass.units.registry
        )
        Tgas_no_agn = unyt.unyt_array(
            Tgas_no_agn, dtype=np.float32, units="K", registry=mass.units.registry
        )

        halfmass = {}
        for name, r, m, M in zip(
            ["tot", "gas", "dm", "star"],
            [
                radius,
                radius[type == "PartType0"],
                radius[type == "PartType1"],
                radius[type == "PartType4"],
            ],
            [mass, mass_gas, mass_dm, mass_star],
            [Mtot, Mgas, Mdm, Mstar],
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
                f"{prefix}/Ngas": (Ngas, "Number of gas particles."),
                f"{prefix}/Ndm": (Ndm, "Number of dark matter particles."),
                f"{prefix}/Nstar": (Nstar, "Number of star particles."),
                f"{prefix}/Nbh": (Nbh, "Number of black hole particles."),
                f"{prefix}/BHlasteventa": (
                    BHlasteventa,
                    "Scale-factor of last AGN event.",
                ),
                f"{prefix}/BHmaxM": (BHmaxM, "Mass of most massive black hole."),
                f"{prefix}/BHmaxID": (BHmaxID, "ID of most massive black hole."),
                f"{prefix}/BHmaxpos": (
                    BHmaxpos,
                    "Position of most massive black hole.",
                ),
                f"{prefix}/BHmaxvel": (
                    BHmaxvel,
                    "Velocity of most massive black hole.",
                ),
                f"{prefix}/BHmaxAR": (
                    BHmaxAR,
                    "Accretion rate of most massive black hole.",
                ),
                f"{prefix}/BHmaxlasteventa": (
                    BHmaxlasteventa,
                    "Scale-factor of last AGN event for most massive black hole.",
                ),
                f"{prefix}/com": (com, "Centre of mass"),
                f"{prefix}/vcom": (vcom, "Centre of mass velocity"),
                f"{prefix}/Lgas": (
                    totLgas,
                    "Total angular momentum of the gas, relative w.r.t. the gas centre of mass.",
                ),
                f"{prefix}/Ldm": (
                    totLdm,
                    "Total angular momentum of the dark matter, relative w.r.t. the dark matter centre of mass.",
                ),
                f"{prefix}/Lstar": (
                    totLstar,
                    "Total angular momentum of the stars, relative w.r.t. the stellar centre of mass.",
                ),
                f"{prefix}/kappa_corot_gas": (gas_kappa_corot, "Kappa corot for gas."),
                f"{prefix}/kappa_corot_star": (
                    star_kappa_corot,
                    "Kappa corot for stars.",
                ),
                f"{prefix}/veldisp_gas": (
                    veldisp_gas,
                    "Velocity dispersion of the gas. Measured relative to the centre of mass velocity of all particles. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
                ),
                f"{prefix}/veldisp_dm": (
                    veldisp_dm,
                    "Velocity dispersion of the dark matter. Measured relative to the centre of mass velocity of all particles. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
                ),
                f"{prefix}/veldisp_star": (
                    veldisp_star,
                    "Velocity dispersion of the stars. Measured relative to the centre of mass velocity of all particles. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
                ),
                f"{prefix}/Mgas_SF": (Mgasmetal, "Total mass of star-forming gas."),
                f"{prefix}/Mgas_noSF": (
                    Mgasmetal,
                    "Total mass of non star-forming gas.",
                ),
                f"{prefix}/Mgasmetal": (Mgasmetal, "Total gas mass in metals."),
                f"{prefix}/Mgasmetal_SF": (
                    Mgasmetal_SFR,
                    "Total gas mass in metals for gas that is star-forming.",
                ),
                f"{prefix}/Mgasmetal_noSF": (
                    Mgasmetal_noSFR,
                    "Total gas mass in metals for gas that is non star-forming.",
                ),
                f"{prefix}/MgasO": (MgasO, "Total gas mass in oxygen."),
                f"{prefix}/MgasO_SF": (
                    MgasO_SFR,
                    "Total gas mass in oxygen for gas that is star-forming.",
                ),
                f"{prefix}/MgasO_noSF": (
                    MgasO_noSFR,
                    "Total gas mass in oxygen for gas that is non star-forming.",
                ),
                f"{prefix}/MgasFe": (MgasFe, "Total gas mass in iron."),
                f"{prefix}/MgasFe_SF": (
                    MgasFe_SFR,
                    "Total gas mass in iron for gas that is star-forming.",
                ),
                f"{prefix}/MgasFe_noSF": (
                    MgasFe_noSFR,
                    "Total gas mass in iron for gas that is non star-forming.",
                ),
                f"{prefix}/Tgas": (Tgas, "Mass-weighted gas temperature."),
                f"{prefix}/Tgas_no_agn": (
                    Tgas,
                    "Mass-weighted gas temperature, excluding gas that was recently heated by AGN.",
                ),
                f"{prefix}/SFR": (SFR, "Total SFR"),
                f"{prefix}/Luminosity": (lum, "Total luminosity"),
                f"{prefix}/Mstarmetal": (Mstarmetal, "Total stellar mass in metals."),
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


class DummyCellGrid:
    def __init__(self):
        self.cosmology = {
            "H0 [internal units]": 68.09999996711613,
            "Omega_b": 0.0486,
            "Omega_lambda": 0.693922,
            "Omega_r": 7.791804710018577e-05,
            "Omega_m": 0.30461099999999997,
            "w_0": -1.0,
            "w_a": 0.0,
            "Redshift": 0.1,
        }
        reg = unyt.unit_registry.UnitRegistry()
        unyt.define_unit("snap_length", 3.08567758e24 * unyt.cm, registry=reg)
        unyt.define_unit("snap_mass", 1.98841e43 * unyt.g, registry=reg)
        unyt.define_unit("snap_time", 3.08567758e19 * unyt.s, registry=reg)
        unyt.define_unit("snap_temperature", 1.0 * unyt.K, registry=reg)
        unyt.define_unit("snap_angle", 1.0 * unyt.rad, registry=reg)
        unyt.define_unit("snap_current", 1.0 * unyt.A, registry=reg)

        us = unyt.UnitSystem(
            "snap_units",
            unyt.Unit("snap_length", registry=reg),
            unyt.Unit("snap_mass", registry=reg),
            unyt.Unit("snap_time", registry=reg),
            unyt.Unit("snap_temperature", registry=reg),
            unyt.Unit("snap_angle", registry=reg),
            unyt.Unit("snap_current", registry=reg),
            registry=reg,
        )
        self.snap_unit_registry = unyt.unit_registry.UnitRegistry(
            lut=reg.lut, unit_system=us
        )


class DummyExclusiveSphereProperties(ExclusiveSphereProperties):
    def __init__(self):

        self.cellgrid = DummyCellGrid()
        self.filter = RecentlyHeatedGasFilter(self.cellgrid)
        self.physical_radius_mpc = 0.001


def test_exclusive_sphere_properties():

    property_calculator = DummyExclusiveSphereProperties()
    reg = property_calculator.cellgrid.snap_unit_registry

    np.random.seed(3589)
    for i in range(100):
        npart = np.random.choice([1, 10, 100, 1000, 10000])

        centre = unyt.unyt_array(
            100.0 * np.random.random(3), dtype=np.float64, units=unyt.Mpc, registry=reg
        )
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
        coords = unyt.unyt_array(coords, dtype=np.float64, units=unyt.kpc, registry=reg)
        coords += centre
        mass = unyt.unyt_array(
            1.0e9 * (1.0 - 0.2 * np.random.random(npart)),
            dtype=np.float32,
            units=unyt.Msun,
            registry=reg,
        )
        vs = unyt.unyt_array(
            200.0 * (np.random.random((npart, 3)) - 0.5),
            dtype=np.float32,
            units=unyt.km / unyt.s,
            registry=reg,
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
            data["PartType0"]["LastAGNFeedbackScaleFactors"] = unyt.unyt_array(
                1.0 / 1.1 + 0.01 * np.random.random(Ngas),
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            data["PartType0"]["Masses"] = mass[gas_mask]
            data["PartType0"]["MetalMassFractions"] = unyt.unyt_array(
                1.0e-4 * np.random.random(Ngas),
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            data["PartType0"]["SmoothedElementMassFractions"] = unyt.unyt_array(
                1.0e-4 * np.random.random((Ngas, 9)),
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            data["PartType0"]["StarFormationRates"] = unyt.unyt_array(
                (np.random.random(Ngas) - 0.5),
                dtype=np.float32,
                units=unyt.Msun / unyt.yr,
                registry=reg,
            )
            data["PartType0"]["Temperatures"] = unyt.unyt_array(
                10.0 ** (10.0 * np.random.random(Ngas)),
                dtype=np.float32,
                units=unyt.K,
                registry=reg,
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
                registry=reg,
            )
            data["PartType4"]["Luminosities"] = unyt.unyt_array(
                np.random.random((Nstar, 9)),
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            data["PartType4"]["Masses"] = mass[star_mask]
            data["PartType4"]["MetalMassFractions"] = unyt.unyt_array(
                1.0e-4 * np.random.random(Nstar),
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            data["PartType4"]["Velocities"] = vs[star_mask]

        bh_mask = types == "PartType5"
        Nbh = int(bh_mask.sum())
        if Nbh > 0:
            data["PartType5"] = {}
            data["PartType5"]["AccretionRates"] = unyt.unyt_array(
                np.random.random(Nbh),
                dtype=np.float32,
                units=unyt.Msun / unyt.yr,
                registry=reg,
            )
            data["PartType5"]["Coordinates"] = coords[bh_mask]
            data["PartType5"]["DynamicalMasses"] = mass[bh_mask]
            data["PartType5"]["GroupNr_bound"] = groupnr[bh_mask]
            data["PartType5"]["LastAGNFeedbackScaleFactors"] = unyt.unyt_array(
                1.0 / 1.1 + 0.01 * np.random.random(Nbh),
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            data["PartType5"]["ParticleIDs"] = unyt.unyt_array(
                np.arange(Nbh, dtype=np.uint64),
                dtype=np.uint64,
                units=unyt.dimensionless,
                registry=reg,
            )
            data["PartType5"]["SubgridMasses"] = unyt.unyt_array(
                mass[bh_mask].value * (0.9 + 0.1 * np.random.random(Nbh)),
                dtype=np.float32,
                units=unyt.Msun,
                registry=reg,
            )
            data["PartType5"]["Velocities"] = vs[bh_mask]

        input_halo = {}
        input_halo["cofp"] = centre
        input_halo["index"] = groupnr_halo

        halo_result = {}

        property_calculator.calculate(input_halo, data, halo_result)

        for name, size, dtype, unit in [
            ("ExclusiveSphere/1kpc/Mtot", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/Mgas", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/Mdm", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/Mstar", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/Mstar_init", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/Mbh", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/Mbh_subgrid", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/Ngas", 1, np.uint32, unyt.dimensionless),
            ("ExclusiveSphere/1kpc/Ndm", 1, np.uint32, unyt.dimensionless),
            ("ExclusiveSphere/1kpc/Nstar", 1, np.uint32, unyt.dimensionless),
            ("ExclusiveSphere/1kpc/Nbh", 1, np.uint32, unyt.dimensionless),
            ("ExclusiveSphere/1kpc/BHlasteventa", 1, np.float32, unyt.dimensionless),
            ("ExclusiveSphere/1kpc/BHmaxM", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/BHmaxID", 1, np.uint64, unyt.dimensionless),
            ("ExclusiveSphere/1kpc/BHmaxpos", 3, np.float64, unyt.kpc),
            ("ExclusiveSphere/1kpc/BHmaxvel", 3, np.float32, unyt.km / unyt.s),
            ("ExclusiveSphere/1kpc/BHmaxAR", 1, np.float32, unyt.Msun / unyt.yr),
            ("ExclusiveSphere/1kpc/BHmaxlasteventa", 1, np.float32, unyt.dimensionless),
            ("ExclusiveSphere/1kpc/com", 3, np.float32, unyt.kpc),
            ("ExclusiveSphere/1kpc/vcom", 3, np.float32, unyt.km / unyt.s),
            (
                "ExclusiveSphere/1kpc/Lgas",
                3,
                np.float64,
                unyt.Msun * unyt.kpc * unyt.km / unyt.s,
            ),
            (
                "ExclusiveSphere/1kpc/Ldm",
                3,
                np.float64,
                unyt.Msun * unyt.kpc * unyt.km / unyt.s,
            ),
            (
                "ExclusiveSphere/1kpc/Lstar",
                3,
                np.float64,
                unyt.Msun * unyt.kpc * unyt.km / unyt.s,
            ),
            ("ExclusiveSphere/1kpc/kappa_corot_gas", 1, np.float32, unyt.dimensionless),
            (
                "ExclusiveSphere/1kpc/kappa_corot_star",
                1,
                np.float32,
                unyt.dimensionless,
            ),
            (
                "ExclusiveSphere/1kpc/veldisp_gas",
                6,
                np.float32,
                unyt.km**2 / unyt.s**2,
            ),
            (
                "ExclusiveSphere/1kpc/veldisp_dm",
                6,
                np.float32,
                unyt.km**2 / unyt.s**2,
            ),
            (
                "ExclusiveSphere/1kpc/veldisp_star",
                6,
                np.float32,
                unyt.km**2 / unyt.s**2,
            ),
            ("ExclusiveSphere/1kpc/Mgas_SF", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/Mgas_noSF", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/Mgasmetal", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/Mgasmetal_SF", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/Mgasmetal_noSF", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/MgasO", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/MgasO_SF", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/MgasO_noSF", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/MgasFe", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/MgasFe_SF", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/MgasFe_noSF", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/Tgas", 1, np.float32, unyt.K),
            ("ExclusiveSphere/1kpc/Tgas_no_agn", 1, np.float32, unyt.K),
            ("ExclusiveSphere/1kpc/SFR", 1, np.float32, unyt.Msun / unyt.yr),
            ("ExclusiveSphere/1kpc/Luminosity", 9, np.float32, unyt.dimensionless),
            ("ExclusiveSphere/1kpc/Mstarmetal", 1, np.float32, unyt.Msun),
            ("ExclusiveSphere/1kpc/HalfMassRadiusTot", 1, np.float32, unyt.kpc),
            ("ExclusiveSphere/1kpc/HalfMassRadiusGas", 1, np.float32, unyt.kpc),
            ("ExclusiveSphere/1kpc/HalfMassRadiusDM", 1, np.float32, unyt.kpc),
            ("ExclusiveSphere/1kpc/HalfMassRadiusStar", 1, np.float32, unyt.kpc),
        ]:
            assert name in halo_result
            result = halo_result[name][0]
            assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
            assert result.dtype == dtype
            assert result.units.same_dimensions_as(unit.units)
