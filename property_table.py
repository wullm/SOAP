#!/bin/env python

import numpy as np
import unyt


class PropertyTable:

    categories = ["basic", "general", "gas", "dm", "star", "baryon"]
    explanation = [
        ["BHmaxM"],
        ["com", "vcom"],
        ["Lgas", "Ldm", "Lstar", "Lbaryons"],
        ["kappa_corot_gas", "kappa_corot_star", "kappa_corot_baryons"],
        ["SFR", "MgasFe_SF", "MgasO_SF", "Mgas_SF", "Mgasmetal_SF"],
        ["Tgas", "Tgas_no_agn", "Tgas_no_cool", "Tgas_no_cool_no_agn"],
        ["StellarLuminosity"],
        ["R_vmax", "Vmax"],
        ["spin_parameter"],
        ["veldisp_matrix_gas", "veldisp_matrix_dm", "veldisp_matrix_star"],
        ["proj_veldisp_gas", "proj_veldisp_dm", "proj_veldisp_star"],
        ["MgasO", "MgasO_SF", "MgasFe", "MgasFe_SF", "Mgasmetal", "Mgasmetal_SF"],
        [
            "HalfMassRadiusTot",
            "HalfMassRadiusGas",
            "HalfMassRadiusDM",
            "HalfMassRadiusStar",
        ],
        ["Mfrac_satellites"],
        ["Ekin_gas", "Ekin_star"],
        ["Etherm_gas"],
        ["Mnu", "MnuNS"],
        ["Xraylum", "Xraylum_no_agn", "Xrayphlum", "Xrayphlum_no_agn"],
        ["compY", "compY_no_agn"],
    ]

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
        ),
        "BHmaxAR": (
            "MostMassiveBlackHoleAccretionRate",
            1,
            np.float32,
            "Msun/yr",
            "Gas accretion rate of most massive black hole.",
            "general",
        ),
        "BHmaxID": (
            "MostMassiveBlackHoleID",
            1,
            np.uint64,
            "dimensionless",
            "ID of most massive black hole.",
            "basic",
        ),
        "BHmaxM": (
            "MostMassiveBlackHoleMass",
            1,
            np.float32,
            "Msun",
            "Mass of most massive black hole.",
            "basic",
        ),
        "BHmaxlasteventa": (
            "MostMassiveBlackHoleLastEventScalefactor",
            1,
            np.float32,
            "dimensionless",
            "Scale-factor of last AGN event for most massive black hole.",
            "general",
        ),
        "BHmaxpos": (
            "MostMassiveBlackHolePosition",
            3,
            np.float64,
            "kpc",
            "Position of most massive black hole.",
            "general",
        ),
        "BHmaxvel": (
            "MostMassiveBlackHoleVelocity",
            3,
            np.float32,
            "km/s",
            "Velocity of most massive black hole relative to the simulation volume.",
            "general",
        ),
        "BaryonAxisLengths": (
            "BaryonAxisLengths",
            3,
            np.float32,
            "kpc",
            "Axis lengths of the baryonic (gas and stars) mass distribution, computed from the 3D baryon inertia tensor, relative to the centre of potential..",
            "baryon",
        ),
        "DMAxisLengths": (
            "DarkMatterAxisLengths",
            3,
            np.float32,
            "kpc",
            "Axis lengths of the dark matter mass distribution, computed from the 3D DM inertia tensor, relative to the centre of potential..",
            "dm",
        ),
        "DiscToTotalMassFraction": (
            "DiscToTotalStellarMassFraction",
            1,
            np.float32,
            "dimensionless",
            "Fraction of the total stellar mass that is in a disc.",
            "star",
        ),
        "DopplerB": ("DopplerB", 1, np.float32, "dimensionless", "Kinetic Sunyaey-Zel'dovich effect.", "gas"),
        "Ekin_gas": (
            "KineticEnergyGas",
            1,
            np.float64,
            "erg",
            "Total kinetic energy of the gas, relative to the gas centre of mass velocity.",
            "gas",
        ),
        "Ekin_star": (
            "KineticEnergyStars",
            1,
            np.float64,
            "erg",
            "Total kinetic energy of the stars, relative to the stellar centre of mass velocity.",
            "star",
        ),
        "Etherm_gas": (
            "ThermalEnergyGas",
            1,
            np.float64,
            "erg",
            "Total thermal energy of the gas.",
            "gas",
        ),
        "GasAxisLengths": (
            "GasAxisLengths",
            3,
            np.float32,
            "kpc",
            "Axis lengths of the gas mass distribution, computed from the 3D gas inertia tensor, relative to the centre of potential..",
            "gas",
        ),
        "HalfMassRadiusBaryon": (
            "HalfMassRadiusBaryons",
            1,
            np.float32,
            "kpc",
            "Baryonic (gas and stars) half mass radius.",
            "baryon",
        ),
        "HalfMassRadiusDM": (
            "HalfMassRadiusDarkMatter",
            1,
            np.float32,
            "kpc",
            "Dark matter half mass radius.",
            "dm",
        ),
        "HalfMassRadiusGas": (
            "HalfMassRadiusGas",
            1,
            np.float32,
            "kpc",
            "Gas half mass radius.",
            "gas",
        ),
        "HalfMassRadiusStar": (
            "HalfMassRadiusStars",
            1,
            np.float32,
            "kpc",
            "Stellar half mass radius.",
            "star",
        ),
        "HalfMassRadiusTot": (
            "HalfMassRadiusTotal",
            1,
            np.float32,
            "kpc",
            "Total half mass radius.",
            "general",
        ),
        "Lbaryons": (
            "AngularMomentumBaryons",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of baryons (gas and stars), relative to the centre of potential and baryonic centre of mass velocity.",
            "baryon",
        ),
        "Ldm": (
            "AngularMomentumDarkMatter",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of the dark matter, relative to the centre of potential and DM centre of mass velocity.",
            "dm",
        ),
        "Lgas": (
            "AngularMomentumGas",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of the gas, relative to the centre of potential and gas centre of mass velocity.",
            "gas",
        ),
        "Lstar": (
            "AngularMomentumStars",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of the stars, relative to the centre of potential and stellar centre of mass velocity.",
            "star",
        ),
        "Mbh_dynamical": (
            "BlackHolesDynamicalMass",
            1,
            np.float32,
            "Msun",
            "Total BH dynamical mass.",
            "basic",
        ),
        "Mbh_subgrid": (
            "BlackHolesSubgridMass",
            1,
            np.float32,
            "Msun",
            "Total BH subgrid mass.",
            "basic",
        ),
        "Mdm": ("DarkMatterMass", 1, np.float32, "Msun", "Total DM mass.", "basic"),
        "Mfrac_satellites": (
            "MassFractionSatellites",
            1,
            np.float32,
            "dimensionless",
            "Fraction of mass that is bound to a satellite.",
            "general",
        ),
        "Mgas": ("GasMass", 1, np.float32, "Msun", "Total gas mass.", "basic"),
        "MgasFe": (
            "GasMassInIron",
            1,
            np.float32,
            "Msun",
            "Total gas mass in iron.",
            "gas",
        ),
        "MgasFe_SF": (
            "StarFormingGasMassInIron",
            1,
            np.float32,
            "Msun",
            "Total gas mass in iron for gas that is star-forming.",
            "gas",
        ),
        "MgasO": (
            "GasMassInOxygen",
            1,
            np.float32,
            "Msun",
            "Total gas mass in oxygen.",
            "gas",
        ),
        "MgasO_SF": (
            "StarFormingGasMassInOxygen",
            1,
            np.float32,
            "Msun",
            "Total gas mass in oxygen for gas that is star-forming.",
            "gas",
        ),
        "Mgas_SF": (
            "StarFormingGasMass",
            1,
            np.float32,
            "Msun",
            "Total mass of star-forming gas.",
            "gas",
        ),
        "Mgasmetal": (
            "GasMassInMetals",
            1,
            np.float32,
            "Msun",
            "Total gas mass in metals.",
            "gas",
        ),
        "Mgasmetal_SF": (
            "StarFormingGasMassInMetals",
            1,
            np.float32,
            "Msun",
            "Total gas mass in metals for gas that is star-forming.",
            "gas",
        ),
        "Mhotgas": (
            "HotGasMass",
            1,
            np.float32,
            "Msun",
            "Total mass of gas with a temperature above 1e5 K.",
            "gas",
        ),
        "Mnu": (
            "RawNeutrinoMass",
            1,
            np.float32,
            "Msun",
            "Total neutrino particle mass.",
            "basic",
        ),
        "MnuNS": (
            "NoiseSuppressedNeutrinoMass",
            1,
            np.float32,
            "Msun",
            "Noise suppressed total neutrino mass.",
            "basic",
        ),
        "Mstar": ("StellarMass", 1, np.float32, "Msun", "Total stellar mass.", "basic"),
        "MstarFe": (
            "StellarMassInIron",
            1,
            np.float32,
            "Msun",
            "Total stellar mass in iron.",
            "star",
        ),
        "MstarO": (
            "StellarMassInOxygen",
            1,
            np.float32,
            "Msun",
            "Total stellar mass in oxygen.",
            "star",
        ),
        "Mstar_init": (
            "StellarInitialMass",
            1,
            np.float32,
            "Msun",
            "Total stellar initial mass.",
            "star",
        ),
        "Mstarmetal": (
            "StellarMassInMetals",
            1,
            np.float32,
            "Msun",
            "Total stellar mass in metals.",
            "star",
        ),
        "Mtot": ("TotalMass", 1, np.float32, "Msun", "Total mass.", "basic"),
        "Nbh": (
            "NumberOfBlackHoleParticles",
            1,
            np.uint32,
            "dimensionless",
            "Number of black hole particles.",
            "basic",
        ),
        "Ndm": (
            "NumberOfDarkMatterParticles",
            1,
            np.uint32,
            "dimensionless",
            "Number of dark matter particles.",
            "basic",
        ),
        "Ngas": (
            "NumberOfGasParticles",
            1,
            np.uint32,
            "dimensionless",
            "Number of gas particles.",
            "basic",
        ),
        "Nnu": (
            "NumberOfNeutrinoParticles",
            1,
            np.uint32,
            "dimensionless",
            "Number of neutrino particles.",
            "basic",
        ),
        "Nstar": (
            "NumberOfStarParticles",
            1,
            np.uint32,
            "dimensionless",
            "Number of star particles.",
            "basic",
        ),
        "R_vmax": (
            "MaximumCircularVelocityRadius",
            1,
            np.float32,
            "kpc",
            "Radius at which Vmax is reached.",
            "general",
        ),
        "SFR": (
            "StarFormationRate",
            1,
            np.float32,
            "Msun/yr",
            "Total star formation rate.",
            "general",
        ),
        "StellarAxisLengths": (
            "StellarAxisLengths",
            3,
            np.float32,
            "kpc",
            "Axis lengths of the stellar mass distribution, computed from the 3D stellar inertia tensor, relative to the centre of potential..",
            "star",
        ),
        "StellarLuminosity": (
            "StellarLuminosity",
            9,
            np.float32,
            "dimensionless",
            "Total stellar luminosity in the 9 GAMA bands.",
            "star",
        ),
        "Tgas": (
            "GasTemperature",
            1,
            np.float32,
            "K",
            "Mass-weighted mean gas temperature.",
            "gas",
        ),
        "Tgas_no_agn": (
            "GasTemperatureWithoutRecentAGNHeating",
            1,
            np.float32,
            "K",
            "Mass-weighted mean gas temperature, excluding gas that was recently heated by AGN.",
            "gas",
        ),
        "Tgas_no_cool": (
            "GasTemperatureWithoutCoolGas",
            1,
            np.float32,
            "K",
            "Mass-weighted mean gas temperature, excluding cool gas with a temperature below 1e5 K.",
            "gas",
        ),
        "Tgas_no_cool_no_agn": (
            "GasTemperatureWithoutCoolGasAndRecentAGNHeating",
            1,
            np.float32,
            "K",
            "Mass-weighted mean gas temperature, excluding cool gas with a temperature below 1e5 K and gas that was recently heated by AGN.",
            "gas",
        ),
        "TotalAxisLengths": (
            "TotalAxisLengths",
            3,
            np.float32,
            "kpc",
            "Axis lengths of the total mass distribution, computed from the 3D inertia tensor, relative to the centre of potential..",
            "general",
        ),
        "Vmax": (
            "MaximumCircularVelocity",
            1,
            np.float32,
            "km/s",
            "Maximum circular velocity.",
            "general",
        ),
        "Xraylum": (
            "XRayLuminosity",
            3,
            np.float64,
            "erg/s",
            "Total rest-frame Xray luminosity in three bands.",
            "gas",
        ),
        "Xraylum_no_agn": (
            "XRayLuminosityWithoutRecentAGNHeating",
            3,
            np.float64,
            "erg/s",
            "Total rest-frame Xray luminosity in three bands. Excludes gas that was recently heated by AGN.",
            "gas",
        ),
        "Xrayphlum": (
            "XRayPhotonLuminosity",
            3,
            np.float64,
            "1/s",
            "Total rest-frame Xray photon luminosity in three bands.",
            "gas",
        ),
        "Xrayphlum_no_agn": (
            "XRayPhotonLuminosityWithoutRecentAGNHeating",
            3,
            np.float64,
            "1/s",
            "Total rest-frame Xray photon luminosity in three bands. Exclude gas that was recently heated by AGN.",
            "gas",
        ),
        "com": ("CentreOfMass", 3, np.float32, "kpc", "Centre of mass.", "basic"),
        "com_gas": (
            "GasCentreOfMass",
            3,
            np.float32,
            "Mpc",
            "Centre of mass of gas.",
            "gas",
        ),
        "com_star": (
            "StellarCentreOfMass",
            3,
            np.float32,
            "Mpc",
            "Centre of mass of stars.",
            "star",
        ),
        "compY": (
            "ComptonY",
            1,
            np.float64,
            "cm**2",
            "Total Compton y parameter.",
            "gas",
        ),
        "compY_no_agn": (
            "ComptonYWithoutRecentAGNHeating",
            1,
            np.float64,
            "cm**2",
            "Total Compton y parameter. Excludes gas that was recently heated by AGN.",
            "gas",
        ),
        "kappa_corot_baryons": (
            "KappaCorotBaryons",
            1,
            np.float32,
            "dimensionless",
            "Kappa-corot for baryons (gas and stars), relative to the centre of potential and the centre of mass velocity of the baryons.",
            "baryon",
        ),
        "kappa_corot_gas": (
            "KappaCorotGas",
            1,
            np.float32,
            "dimensionless",
            "Kappa-corot for gas, relative to the centre of potential and the centre of mass velocity of the gas.",
            "gas",
        ),
        "kappa_corot_star": (
            "KappaCorotStars",
            1,
            np.float32,
            "dimensionless",
            "Kappa-corot for stars, relative to the centre of potential and the centre of mass velocity of the stars.",
            "star",
        ),
        "proj_veldisp_dm": (
            "DarkMatterProjectedVelocityDispersion",
            1,
            np.float32,
            "km/s",
            "Mass-weighted velocity dispersion of the DM along the projection axis, relative to the DM centre of mass velocity.",
            "dm",
        ),
        "proj_veldisp_gas": (
            "GasProjectedVelocityDispersion",
            1,
            np.float32,
            "km/s",
            "Mass-weighted velocity dispersion of the gas along the projection axis, relative to the gas centre of mass velocity.",
            "gas",
        ),
        "proj_veldisp_star": (
            "StellarProjectedVelocityDispersion",
            1,
            np.float32,
            "km/s",
            "Mass-weighted velocity dispersion of the stars along the projection axis, relative to the stellar centre of mass velocity.",
            "star",
        ),
        "r": ("SORadius", 1, np.float32, "Mpc", "Radius of a sphere {label}", "basic"),
        "spin_parameter": (
            "SpinParameter",
            1,
            np.float32,
            "dimensionless",
            "Bullock et al. (2001) spin parameter.",
            "general",
        ),
        "vcom": (
            "CentreOfMassVelocity",
            3,
            np.float32,
            "km/s",
            "Centre of mass velocity.",
            "basic",
        ),
        "vcom_gas": (
            "GasCentreOfMassVelocity",
            3,
            np.float32,
            "km/s",
            "Centre of mass velocity of gas.",
            "gas",
        ),
        "vcom_star": (
            "StellarCentreOfMassVelocity",
            3,
            np.float32,
            "km/s",
            "Centre of mass velocity of stars.",
            "star",
        ),
        "veldisp_matrix_dm": (
            "DarkMatterVelocityDispersionMatrix",
            6,
            np.float32,
            "km**2/s**2",
            "Mass-weighted velocity dispersion of the dark matter. Measured relative to the DM centre of mass velocity. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
            "dm",
        ),
        "veldisp_matrix_gas": (
            "GasVelocityDispersionMatrix",
            6,
            np.float32,
            "km**2/s**2",
            "Mass-weighted velocity dispersion of the gas. Measured relative to the gas centre of mass velocity. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
            "gas",
        ),
        "veldisp_matrix_star": (
            "StellarVelocityDispersionMatrix",
            6,
            np.float32,
            "km**2/s**2",
            "Mass-weighted velocity dispersion of the stars. Measured relative to the stellar centre of mass velocity. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
            "star",
        ),
    }

    def get_footnotes(self, name):
        footnotes = []
        for i, names in enumerate(self.explanation):
            if name in names:
                footnotes.append(i + 1)
        if len(footnotes) > 0:
            return f'$^{{{",".join([f"{i}" for i in footnotes])}}}$'
        else:
            return ""

    def __init__(self):
        self.properties = {}

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
                raw_shape,
                raw_dtype,
                raw_units,
                raw_description,
                raw_cat,
            ) = self.properties[name]["raw"]
            raw_dtype = f"np.{raw_dtype.__name__}"
            print(
                f'  "{raw_name}": ("{raw_name}", {raw_shape}, {raw_dtype}, "{raw_units}", "{raw_description}", "{raw_cat}"),'
            )
        print("}")

    def print_table(self):
        prop_names = sorted(
            self.properties.keys(),
            key=lambda key: (
                self.categories.index(self.properties[key]["category"]),
                self.properties[key]["name"].lower(),
            ),
        )
        tablestr = """\\documentclass{article}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{longtable}
\\usepackage{pifont}
\\usepackage{pdflscape}
\\usepackage{a4wide}
\\usepackage{multirow}
\\begin{document}

\\begin{landscape}
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
\\end{landscape}
\\end{document}"""
        print(tablestr)


class DummyProperties:
    property_list = [
        (prop, *PropertyTable.full_property_list[prop])
        for prop in PropertyTable.full_property_list.keys()
    ]


if __name__ == "__main__":

    from exclusive_sphere_properties import ExclusiveSphereProperties
    from inclusive_sphere_properties import InclusiveSphereProperties
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
        table.print_table()
