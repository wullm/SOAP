#!/bin/env python

import numpy as np
import unyt


class PropertyTable:

    categories = ["basic", "general", "gas", "dm", "star", "baryon"]
    explanation = {
        "footnote_MBH.tex": ["BHmaxM"],
        "footnote_com.tex": ["com", "vcom"],
        "footnote_AngMom.tex": ["Lgas", "Ldm", "Lstar", "Lbaryons"],
        "footnote_kappa.tex": [
            "kappa_corot_gas",
            "kappa_corot_star",
            "kappa_corot_baryons",
        ],
        "footnote_SF.tex": ["SFR", "MgasFe_SF", "MgasO_SF", "Mgas_SF", "Mgasmetal_SF"],
        "footnote_Tgas.tex": [
            "Tgas",
            "Tgas_no_agn",
            "Tgas_no_cool",
            "Tgas_no_cool_no_agn",
        ],
        "footnote_lum.tex": ["StellarLuminosity"],
        "footnote_circvel.tex": ["R_vmax", "Vmax"],
        "footnote_spin.tex": ["spin_parameter"],
        "footnote_veldisp_matrix.tex": [
            "veldisp_matrix_gas",
            "veldisp_matrix_dm",
            "veldisp_matrix_star",
        ],
        "footnote_proj_veldisp.tex": [
            "proj_veldisp_gas",
            "proj_veldisp_dm",
            "proj_veldisp_star",
        ],
        "footnote_elements.tex": [
            "MgasO",
            "MgasO_SF",
            "MgasFe",
            "MgasFe_SF",
            "Mgasmetal",
            "Mgasmetal_SF",
        ],
        "footnote_halfmass.tex": [
            "HalfMassRadiusTot",
            "HalfMassRadiusGas",
            "HalfMassRadiusDM",
            "HalfMassRadiusStar",
        ],
        "footnote_satfrac.tex": ["Mfrac_satellites"],
        "footnote_Ekin.tex": ["Ekin_gas", "Ekin_star"],
        "footnote_Etherm.tex": ["Etherm_gas"],
        "footnote_Mnu.tex": ["Mnu", "MnuNS"],
        "footnote_Xray.tex": [
            "Xraylum",
            "Xraylum_no_agn",
            "Xrayphlum",
            "Xrayphlum_no_agn",
        ],
        "footnote_compY.tex": ["compY", "compY_no_agn"],
        "footnote_dopplerB.tex": ["DopplerB"],
    }

    # List of properties that get computed
    # For each property, we have the following columns:
    #  - name: Name of the property within calculate() and in the output file
    #  - shape: Shape of this property for a single halo (1: scalar, 3: vector...)
    #  - dtype: Data type that will be used. Should have enough precision to avoid over/underflow
    #  - unit: Units that will be used internally and for the output.
    #  - description: Description string that will be used to describe the property in the output.
    full_property_list = {
        "BHlasteventa": (
            "BlackHolesLastEventScalefactor",
            1,
            np.float32,
            "dimensionless",
            "Scale-factor of last AGN event.",
            "general",
            "FMantissa9",
        ),
        "BHmaxAR": (
            "MostMassiveBlackHoleAccretionRate",
            1,
            np.float32,
            "Msun/yr",
            "Gas accretion rate of most massive black hole.",
            "general",
            "FMantissa9",
        ),
        "BHmaxID": (
            "MostMassiveBlackHoleID",
            1,
            np.uint64,
            "dimensionless",
            "ID of most massive black hole.",
            "basic",
            "FMantissa9",
        ),
        "BHmaxM": (
            "MostMassiveBlackHoleMass",
            1,
            np.float32,
            "Msun",
            "Mass of most massive black hole.",
            "basic",
            "FMantissa9",
        ),
        "BHmaxlasteventa": (
            "MostMassiveBlackHoleLastEventScalefactor",
            1,
            np.float32,
            "dimensionless",
            "Scale-factor of last AGN event for most massive black hole.",
            "general",
            "FMantissa9",
        ),
        "BHmaxpos": (
            "MostMassiveBlackHolePosition",
            3,
            np.float64,
            "kpc",
            "Position of most massive black hole.",
            "general",
            "FMantissa9",
        ),
        "BHmaxvel": (
            "MostMassiveBlackHoleVelocity",
            3,
            np.float32,
            "km/s",
            "Velocity of most massive black hole relative to the simulation volume.",
            "general",
            "FMantissa9",
        ),
        "BaryonAxisLengths": (
            "BaryonAxisLengths",
            3,
            np.float32,
            "kpc",
            "Axis lengths of the baryonic (gas and stars) mass distribution, computed from the 3D baryon inertia tensor, relative to the centre of potential.",
            "baryon",
            "FMantissa9",
        ),
        "DMAxisLengths": (
            "DarkMatterAxisLengths",
            3,
            np.float32,
            "kpc",
            "Axis lengths of the dark matter mass distribution, computed from the 3D DM inertia tensor, relative to the centre of potential.",
            "dm",
            "FMantissa9",
        ),
        "DopplerB": (
            "DopplerB",
            1,
            np.float32,
            "dimensionless",
            "Kinetic Sunyaey-Zel'dovich effect, assuming a line of sight towards the position of the first lightcone observer.",
            "gas",
            "FMantissa9",
        ),
        "DtoTgas": (
            "DiscToTotalGasMassFraction",
            1,
            np.float32,
            "dimensionless",
            "Fraction of the total gas mass that is co-rotating.",
            "gas",
            "FMantissa9",
        ),
        "DtoTstar": (
            "DiscToTotalStellarMassFraction",
            1,
            np.float32,
            "dimensionless",
            "Fraction of the total stellar mass that is co-rotating.",
            "star",
            "FMantissa9",
        ),
        "Ekin_gas": (
            "KineticEnergyGas",
            1,
            np.float64,
            "erg",
            "Total kinetic energy of the gas, relative to the gas centre of mass velocity.",
            "gas",
            "FMantissa9",
        ),
        "Ekin_star": (
            "KineticEnergyStars",
            1,
            np.float64,
            "erg",
            "Total kinetic energy of the stars, relative to the stellar centre of mass velocity.",
            "star",
            "FMantissa9",
        ),
        "Etherm_gas": (
            "ThermalEnergyGas",
            1,
            np.float64,
            "erg",
            "Total thermal energy of the gas.",
            "gas",
            "FMantissa9",
        ),
        "GasAxisLengths": (
            "GasAxisLengths",
            3,
            np.float32,
            "kpc",
            "Axis lengths of the gas mass distribution, computed from the 3D gas inertia tensor, relative to the centre of potential.",
            "gas",
            "FMantissa9",
        ),
        "HalfMassRadiusBaryon": (
            "HalfMassRadiusBaryons",
            1,
            np.float32,
            "kpc",
            "Baryonic (gas and stars) half mass radius.",
            "baryon",
            "FMantissa9",
        ),
        "HalfMassRadiusDM": (
            "HalfMassRadiusDarkMatter",
            1,
            np.float32,
            "kpc",
            "Dark matter half mass radius.",
            "dm",
            "FMantissa9",
        ),
        "HalfMassRadiusGas": (
            "HalfMassRadiusGas",
            1,
            np.float32,
            "kpc",
            "Gas half mass radius.",
            "gas",
            "FMantissa9",
        ),
        "HalfMassRadiusStar": (
            "HalfMassRadiusStars",
            1,
            np.float32,
            "kpc",
            "Stellar half mass radius.",
            "star",
            "FMantissa9",
        ),
        "HalfMassRadiusTot": (
            "HalfMassRadiusTotal",
            1,
            np.float32,
            "kpc",
            "Total half mass radius.",
            "general",
            "FMantissa9",
        ),
        "Lbaryons": (
            "AngularMomentumBaryons",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of baryons (gas and stars), relative to the centre of potential and baryonic centre of mass velocity.",
            "baryon",
            "FMantissa9",
        ),
        "Ldm": (
            "AngularMomentumDarkMatter",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of the dark matter, relative to the centre of potential and DM centre of mass velocity.",
            "dm",
            "FMantissa9",
        ),
        "Lgas": (
            "AngularMomentumGas",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of the gas, relative to the centre of potential and gas centre of mass velocity.",
            "gas",
            "FMantissa9",
        ),
        "Lstar": (
            "AngularMomentumStars",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of the stars, relative to the centre of potential and stellar centre of mass velocity.",
            "star",
            "FMantissa9",
        ),
        "Mbh_dynamical": (
            "BlackHolesDynamicalMass",
            1,
            np.float32,
            "Msun",
            "Total BH dynamical mass.",
            "basic",
            "FMantissa9",
        ),
        "Mbh_subgrid": (
            "BlackHolesSubgridMass",
            1,
            np.float32,
            "Msun",
            "Total BH subgrid mass.",
            "basic",
            "FMantissa9",
        ),
        "Mdm": (
            "DarkMatterMass",
            1,
            np.float32,
            "Msun",
            "Total DM mass.",
            "basic",
            "FMantissa9",
        ),
        "Mfrac_satellites": (
            "MassFractionSatellites",
            1,
            np.float32,
            "dimensionless",
            "Fraction of mass that is bound to a satellite.",
            "general",
            "FMantissa9",
        ),
        "Mgas": (
            "GasMass",
            1,
            np.float32,
            "Msun",
            "Total gas mass.",
            "basic",
            "FMantissa9",
        ),
        "MgasFe": (
            "GasMassInIron",
            1,
            np.float32,
            "Msun",
            "Total gas mass in iron.",
            "gas",
            "FMantissa9",
        ),
        "MgasFe_SF": (
            "StarFormingGasMassInIron",
            1,
            np.float32,
            "Msun",
            "Total gas mass in iron for gas that is star-forming.",
            "gas",
            "FMantissa9",
        ),
        "MgasO": (
            "GasMassInOxygen",
            1,
            np.float32,
            "Msun",
            "Total gas mass in oxygen.",
            "gas",
            "FMantissa9",
        ),
        "MgasO_SF": (
            "StarFormingGasMassInOxygen",
            1,
            np.float32,
            "Msun",
            "Total gas mass in oxygen for gas that is star-forming.",
            "gas",
            "FMantissa9",
        ),
        "Mgas_SF": (
            "StarFormingGasMass",
            1,
            np.float32,
            "Msun",
            "Total mass of star-forming gas.",
            "gas",
            "FMantissa9",
        ),
        "Mgasmetal": (
            "GasMassInMetals",
            1,
            np.float32,
            "Msun",
            "Total gas mass in metals.",
            "gas",
            "FMantissa9",
        ),
        "Mgasmetal_SF": (
            "StarFormingGasMassInMetals",
            1,
            np.float32,
            "Msun",
            "Total gas mass in metals for gas that is star-forming.",
            "gas",
            "FMantissa9",
        ),
        "Mhotgas": (
            "HotGasMass",
            1,
            np.float32,
            "Msun",
            "Total mass of gas with a temperature above 1e5 K.",
            "gas",
            "FMantissa9",
        ),
        "Mnu": (
            "RawNeutrinoMass",
            1,
            np.float32,
            "Msun",
            "Total neutrino particle mass.",
            "basic",
            "FMantissa9",
        ),
        "MnuNS": (
            "NoiseSuppressedNeutrinoMass",
            1,
            np.float32,
            "Msun",
            "Noise suppressed total neutrino mass.",
            "basic",
            "FMantissa9",
        ),
        "Mstar": (
            "StellarMass",
            1,
            np.float32,
            "Msun",
            "Total stellar mass.",
            "basic",
            "FMantissa9",
        ),
        "MstarFe": (
            "StellarMassInIron",
            1,
            np.float32,
            "Msun",
            "Total stellar mass in iron.",
            "star",
            "FMantissa9",
        ),
        "MstarO": (
            "StellarMassInOxygen",
            1,
            np.float32,
            "Msun",
            "Total stellar mass in oxygen.",
            "star",
            "FMantissa9",
        ),
        "Mstar_init": (
            "StellarInitialMass",
            1,
            np.float32,
            "Msun",
            "Total stellar initial mass.",
            "star",
            "FMantissa9",
        ),
        "Mstarmetal": (
            "StellarMassInMetals",
            1,
            np.float32,
            "Msun",
            "Total stellar mass in metals.",
            "star",
            "FMantissa9",
        ),
        "Mtot": (
            "TotalMass",
            1,
            np.float32,
            "Msun",
            "Total mass.",
            "basic",
            "FMantissa9",
        ),
        "Nbh": (
            "NumberOfBlackHoleParticles",
            1,
            np.uint32,
            "dimensionless",
            "Number of black hole particles.",
            "basic",
            "FMantissa9",
        ),
        "Ndm": (
            "NumberOfDarkMatterParticles",
            1,
            np.uint32,
            "dimensionless",
            "Number of dark matter particles.",
            "basic",
            "FMantissa9",
        ),
        "Ngas": (
            "NumberOfGasParticles",
            1,
            np.uint32,
            "dimensionless",
            "Number of gas particles.",
            "basic",
            "FMantissa9",
        ),
        "Nnu": (
            "NumberOfNeutrinoParticles",
            1,
            np.uint32,
            "dimensionless",
            "Number of neutrino particles.",
            "basic",
            "FMantissa9",
        ),
        "Nstar": (
            "NumberOfStarParticles",
            1,
            np.uint32,
            "dimensionless",
            "Number of star particles.",
            "basic",
            "FMantissa9",
        ),
        "ProjectedBaryonAxisLengths": (
            "ProjectedBaryonAxisLengths",
            2,
            np.float32,
            "kpc",
            "Axis lengths of the projected baryon (gas and stars) mass distribution, computed from the 2D baryon inertia tensor, relative to the centre of potential.",
            "baryon",
            "FMantissa9",
        ),
        "ProjectedGasAxisLengths": (
            "ProjectedGasAxisLengths",
            2,
            np.float32,
            "kpc",
            "Axis lengths of the projected gas mass distribution, computed from the 2D gas inertia tensor, relative to the centre of potential.",
            "gas",
            "FMantissa9",
        ),
        "ProjectedStellarAxisLengths": (
            "ProjectedStellarAxisLengths",
            2,
            np.float32,
            "kpc",
            "Axis lengths of the projected stellar mass distribution, computed from the 2D stellar inertia tensor, relative to the centre of potential.",
            "star",
            "FMantissa9",
        ),
        "R_vmax": (
            "MaximumCircularVelocityRadius",
            1,
            np.float32,
            "kpc",
            "Radius at which Vmax is reached.",
            "general",
            "FMantissa9",
        ),
        "SFR": (
            "StarFormationRate",
            1,
            np.float32,
            "Msun/yr",
            "Total star formation rate.",
            "general",
            "FMantissa9",
        ),
        "StellarAxisLengths": (
            "StellarAxisLengths",
            3,
            np.float32,
            "kpc",
            "Axis lengths of the stellar mass distribution, computed from the 3D stellar inertia tensor, relative to the centre of potential.",
            "star",
            "FMantissa9",
        ),
        "StellarLuminosity": (
            "StellarLuminosity",
            9,
            np.float32,
            "dimensionless",
            "Total stellar luminosity in the 9 GAMA bands.",
            "star",
            "FMantissa9",
        ),
        "Tgas": (
            "GasTemperature",
            1,
            np.float32,
            "K",
            "Mass-weighted mean gas temperature.",
            "gas",
            "FMantissa9",
        ),
        "Tgas_no_agn": (
            "GasTemperatureWithoutRecentAGNHeating",
            1,
            np.float32,
            "K",
            "Mass-weighted mean gas temperature, excluding gas that was recently heated by AGN.",
            "gas",
            "FMantissa9",
        ),
        "Tgas_no_cool": (
            "GasTemperatureWithoutCoolGas",
            1,
            np.float32,
            "K",
            "Mass-weighted mean gas temperature, excluding cool gas with a temperature below 1e5 K.",
            "gas",
            "FMantissa9",
        ),
        "Tgas_no_cool_no_agn": (
            "GasTemperatureWithoutCoolGasAndRecentAGNHeating",
            1,
            np.float32,
            "K",
            "Mass-weighted mean gas temperature, excluding cool gas with a temperature below 1e5 K and gas that was recently heated by AGN.",
            "gas",
            "FMantissa9",
        ),
        "TotalAxisLengths": (
            "TotalAxisLengths",
            3,
            np.float32,
            "kpc",
            "Axis lengths of the total mass distribution, computed from the 3D inertia tensor, relative to the centre of potential.",
            "general",
            "FMantissa9",
        ),
        "Vmax": (
            "MaximumCircularVelocity",
            1,
            np.float32,
            "km/s",
            "Maximum circular velocity.",
            "general",
            "FMantissa9",
        ),
        "Xraylum": (
            "XRayLuminosity",
            3,
            np.float64,
            "erg/s",
            "Total rest-frame Xray luminosity in three bands.",
            "gas",
            "FMantissa9",
        ),
        "Xraylum_no_agn": (
            "XRayLuminosityWithoutRecentAGNHeating",
            3,
            np.float64,
            "erg/s",
            "Total rest-frame Xray luminosity in three bands. Excludes gas that was recently heated by AGN.",
            "gas",
            "FMantissa9",
        ),
        "Xrayphlum": (
            "XRayPhotonLuminosity",
            3,
            np.float64,
            "1/s",
            "Total rest-frame Xray photon luminosity in three bands.",
            "gas",
            "FMantissa9",
        ),
        "Xrayphlum_no_agn": (
            "XRayPhotonLuminosityWithoutRecentAGNHeating",
            3,
            np.float64,
            "1/s",
            "Total rest-frame Xray photon luminosity in three bands. Exclude gas that was recently heated by AGN.",
            "gas",
            "FMantissa9",
        ),
        "com": (
            "CentreOfMass",
            3,
            np.float32,
            "kpc",
            "Centre of mass.",
            "basic",
            "FMantissa9",
        ),
        "com_gas": (
            "GasCentreOfMass",
            3,
            np.float32,
            "Mpc",
            "Centre of mass of gas.",
            "gas",
            "FMantissa9",
        ),
        "com_star": (
            "StellarCentreOfMass",
            3,
            np.float32,
            "Mpc",
            "Centre of mass of stars.",
            "star",
            "FMantissa9",
        ),
        "compY": (
            "ComptonY",
            1,
            np.float64,
            "cm**2",
            "Total Compton y parameter.",
            "gas",
            "FMantissa9",
        ),
        "compY_no_agn": (
            "ComptonYWithoutRecentAGNHeating",
            1,
            np.float64,
            "cm**2",
            "Total Compton y parameter. Excludes gas that was recently heated by AGN.",
            "gas",
            "FMantissa9",
        ),
        "kappa_corot_baryons": (
            "KappaCorotBaryons",
            1,
            np.float32,
            "dimensionless",
            "Kappa-corot for baryons (gas and stars), relative to the centre of potential and the centre of mass velocity of the baryons.",
            "baryon",
            "FMantissa9",
        ),
        "kappa_corot_gas": (
            "KappaCorotGas",
            1,
            np.float32,
            "dimensionless",
            "Kappa-corot for gas, relative to the centre of potential and the centre of mass velocity of the gas.",
            "gas",
            "FMantissa9",
        ),
        "kappa_corot_star": (
            "KappaCorotStars",
            1,
            np.float32,
            "dimensionless",
            "Kappa-corot for stars, relative to the centre of potential and the centre of mass velocity of the stars.",
            "star",
            "FMantissa9",
        ),
        "proj_veldisp_dm": (
            "DarkMatterProjectedVelocityDispersion",
            1,
            np.float32,
            "km/s",
            "Mass-weighted velocity dispersion of the DM along the projection axis, relative to the DM centre of mass velocity.",
            "dm",
            "FMantissa9",
        ),
        "proj_veldisp_gas": (
            "GasProjectedVelocityDispersion",
            1,
            np.float32,
            "km/s",
            "Mass-weighted velocity dispersion of the gas along the projection axis, relative to the gas centre of mass velocity.",
            "gas",
            "FMantissa9",
        ),
        "proj_veldisp_star": (
            "StellarProjectedVelocityDispersion",
            1,
            np.float32,
            "km/s",
            "Mass-weighted velocity dispersion of the stars along the projection axis, relative to the stellar centre of mass velocity.",
            "star",
            "FMantissa9",
        ),
        "r": (
            "SORadius",
            1,
            np.float32,
            "Mpc",
            "Radius of a sphere {label}",
            "basic",
            "FMantissa9",
        ),
        "spin_parameter": (
            "SpinParameter",
            1,
            np.float32,
            "dimensionless",
            "Bullock et al. (2001) spin parameter.",
            "general",
            "FMantissa9",
        ),
        "stellar_age_lw": (
            "LuminosityWeightedMeanStellarAge",
            1,
            np.float32,
            "Myr",
            "Luminosity weighted mean stellar age. The weight is the r band luminosity.",
            "star",
            "FMantissa9",
        ),
        "stellar_age_mw": (
            "MassWeightedMeanStellarAge",
            1,
            np.float32,
            "Myr",
            "Mass weighted mean stellar age.",
            "star",
            "FMantissa9",
        ),
        "vcom": (
            "CentreOfMassVelocity",
            3,
            np.float32,
            "km/s",
            "Centre of mass velocity.",
            "basic",
            "FMantissa9",
        ),
        "vcom_gas": (
            "GasCentreOfMassVelocity",
            3,
            np.float32,
            "km/s",
            "Centre of mass velocity of gas.",
            "gas",
            "FMantissa9",
        ),
        "vcom_star": (
            "StellarCentreOfMassVelocity",
            3,
            np.float32,
            "km/s",
            "Centre of mass velocity of stars.",
            "star",
            "FMantissa9",
        ),
        "veldisp_matrix_dm": (
            "DarkMatterVelocityDispersionMatrix",
            6,
            np.float32,
            "km**2/s**2",
            "Mass-weighted velocity dispersion of the dark matter. Measured relative to the DM centre of mass velocity. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
            "dm",
            "FMantissa9",
        ),
        "veldisp_matrix_gas": (
            "GasVelocityDispersionMatrix",
            6,
            np.float32,
            "km**2/s**2",
            "Mass-weighted velocity dispersion of the gas. Measured relative to the gas centre of mass velocity. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
            "gas",
            "FMantissa9",
        ),
        "veldisp_matrix_star": (
            "StellarVelocityDispersionMatrix",
            6,
            np.float32,
            "km**2/s**2",
            "Mass-weighted velocity dispersion of the stars. Measured relative to the stellar centre of mass velocity. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
            "star",
            "FMantissa9",
        ),
    }

    def get_footnotes(self, name):
        footnotes = []
        for fnote in self.explanation.keys():
            names = self.explanation[fnote]
            if name in names:
                try:
                    i = self.footnotes.index(fnote)
                except ValueError:
                    i = len(self.footnotes)
                    self.footnotes.append(fnote)
                footnotes.append(i + 1)
        if len(footnotes) > 0:
            return f'$^{{{",".join([f"{i}" for i in footnotes])}}}$'
        else:
            return ""

    def __init__(self):
        self.properties = {}
        self.footnotes = []

    def add_properties(self, halo_property):
        halo_type = halo_property.__name__
        props = halo_property.property_list
        for i, (
            prop_name,
            prop_outputname,
            prop_shape,
            prop_dtype,
            prop_units,
            prop_description,
            prop_cat,
            prop_comp,
        ) in enumerate(props):
            prop_units = (
                unyt.unyt_quantity(1, units=prop_units)
                .units.latex_repr.replace(
                    "\\rm{km} \\cdot \\rm{kpc}", "\\rm{kpc} \\cdot \\rm{km}"
                )
                .replace(
                    "\\frac{\\rm{km}^{2}}{\\rm{s}^{2}}", "\\rm{km}^{2} / \\rm{s}^{2}"
                )
            )
            prop_dtype = prop_dtype.__name__
            if prop_name in self.properties:
                if not prop_cat in self.categories:
                    print(f"Unknown category: {prop_cat}!")
                    exit()
                # run some checks
                if prop_shape != self.properties[prop_name]["shape"]:
                    print("Shape mismatch!")
                    print(halo_type, prop_name, prop_shape, self.properties[prop_name])
                    exit()
                if prop_dtype != self.properties[prop_name]["dtype"]:
                    print("dtype mismatch!")
                    print(halo_type, prop_name, prop_dtype, self.properties[prop_name])
                    exit()
                if prop_units != self.properties[prop_name]["units"]:
                    print("Unit mismatch!")
                    print(halo_type, prop_name, prop_units, self.properties[prop_name])
                    exit()
                if prop_description != self.properties[prop_name]["description"]:
                    print("Description mismatch!")
                    print(
                        halo_type,
                        prop_name,
                        prop_description,
                        self.properties[prop_name],
                    )
                    exit()
                if prop_cat != self.properties[prop_name]["category"]:
                    print("Category mismatch!")
                    print(halo_type, prop_name, prop_cat, self.properties[prop_name])
                    exit()
                assert prop_outputname == self.properties[prop_name]["name"]
                self.properties[prop_name]["types"].append(halo_type)
            else:
                self.properties[prop_name] = {
                    "name": prop_outputname,
                    "shape": prop_shape,
                    "dtype": prop_dtype,
                    "units": prop_units,
                    "description": prop_description,
                    "category": prop_cat,
                    "types": [halo_type],
                    "raw": props[i],
                }

    def print_dictionary(self):
        names = sorted(list(self.properties.keys()))
        print("full_property_list = {")
        for name in names:
            (
                raw_name,
                raw_outputname,
                raw_shape,
                raw_dtype,
                raw_units,
                raw_description,
                raw_cat,
                raw_comp,
            ) = self.properties[name]["raw"]
            raw_dtype = f"np.{raw_dtype.__name__}"
            print(
                f'  "{raw_name}": ("{raw_outputname}", {raw_shape}, {raw_dtype}, "{raw_units}", "{raw_description}", "{raw_cat}", "{raw_comp}"),'
            )
        print("}")

    def print_table(self, tablefile, footnotefile):
        prop_names = sorted(
            self.properties.keys(),
            key=lambda key: (
                self.categories.index(self.properties[key]["category"]),
                self.properties[key]["name"].lower(),
            ),
        )
        headstr = """\\documentclass{article}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{longtable}
\\usepackage{pifont}
\\usepackage{pdflscape}
\\usepackage{a4wide}
\\usepackage{multirow}
\\begin{document}"""

        tablestr = """\\begin{landscape}
\\begin{longtable}{llllllllll}
Name & Shape & Type & Units & SH & ES & IS & EP & SO & Category \\\\
\\multicolumn{10}{l}{\\rule{30pt}{0pt}Description}\\\\
\\hline{}\\endhead{}"""
        prev_cat = None
        for prop_name in prop_names:
            prop = self.properties[prop_name]
            footnotes = self.get_footnotes(prop_name)
            prop_outputname = f"{prop['name'].replace('_','')}{footnotes}"
            prop_shape = f'{prop["shape"]}'
            prop_dtype = prop["dtype"]
            prop_units = f'${prop["units"]}$' if prop["units"] != "" else "(no unit)"
            prop_cat = prop["category"]
            prop_description = prop["description"].format(
                label="satisfying a spherical overdensity criterion."
            )
            checkmark = "\\ding{51}"
            xmark = "\\ding{53}"
            prop_subhalo = checkmark if "SubhaloProperties" in prop["types"] else xmark
            prop_exclusive = (
                checkmark if "ExclusiveSphereProperties" in prop["types"] else xmark
            )
            prop_inclusive = (
                checkmark if "InclusiveSphereProperties" in prop["types"] else xmark
            )
            prop_projected = (
                checkmark if "ProjectedApertureProperties" in prop["types"] else xmark
            )
            prop_SO = checkmark if "SOProperties" in prop["types"] else xmark
            if prev_cat is None:
                prev_cat = prop_cat
            if prop_cat != prev_cat:
                prev_cat = prop_cat
                tablestr += "\\hline{}"
            tablestr += (
                "\\rule{0pt}{4ex}"
                + " & ".join(
                    [
                        v
                        for v in [
                            prop_outputname,
                            prop_shape,
                            prop_dtype,
                            prop_units,
                            prop_subhalo,
                            prop_exclusive,
                            prop_inclusive,
                            prop_projected,
                            prop_SO,
                            prop_cat,
                        ]
                    ]
                )
                + "\\\\*\n"
            )
            tablestr += f"\\multicolumn{{10}}{{p{{20cm}}}}{{\\rule{{30pt}}{{0pt}}{prop_description}}}\\\\\n"
        tablestr += """\\end{longtable}
\\end{landscape}"""
        tailstr = "\\end{document}"
        with open(tablefile, "w") as ofile:
            ofile.write(tablestr)
        with open(footnotefile, "w") as ofile:
            for i, fnote in enumerate(self.footnotes):
                with open(f"documentation/{fnote}", "r") as ifile:
                    fnstr = ifile.read()
                fnstr = fnstr.replace("$FOOTNOTE_NUMBER$", f"{i+1}")
                ofile.write(f"{fnstr}\n\n")
        print(f"{headstr}\n{tablestr}\n{tailstr}")


class DummyProperties:
    property_list = [
        (prop, *PropertyTable.full_property_list[prop])
        for prop in PropertyTable.full_property_list.keys()
    ]


if __name__ == "__main__":

    from aperture_properties import ExclusiveSphereProperties, InclusiveSphereProperties
    from projected_aperture_properties import ProjectedApertureProperties
    from SO_properties import SOProperties
    from subhalo_properties import SubhaloProperties

    table = PropertyTable()
    table.add_properties(ExclusiveSphereProperties)
    table.add_properties(InclusiveSphereProperties)
    table.add_properties(ProjectedApertureProperties)
    table.add_properties(SOProperties)
    table.add_properties(SubhaloProperties)
    table.add_properties(DummyProperties)

    if False:
        table.print_dictionary()
    else:
        table.print_table("documentation/table.tex", "documentation/footnotes.tex")
