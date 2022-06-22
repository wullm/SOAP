#! /usr/bin/env python3

import numpy as np
import unyt


class DummyCellGrid:
    """
    Minimal CellGrid, that contains just enough information to be passed on to
    the RecentlyHeatedGasFilter constructor.
    """

    def __init__(self, reg):
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
        self.snap_unit_registry = reg


class DummyHaloGenerator:
    """
    Object used to generate random halos.

    The random halos contain all the variables a real halo would get, expressed
    in the right units and with realistic values.
    """

    def __init__(self, seed):
        """
        Set up an artificial snapshot unit system.
        """

        # set up snapshot simulation units
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
        self.unit_registry = unyt.unit_registry.UnitRegistry(
            lut=reg.lut, unit_system=us
        )

        np.random.seed(seed)

    def get_cell_grid(self):
        """
        Return a minimal cell grid that is consistent with the random halos
        that are generated.
        """
        return DummyCellGrid(self.unit_registry)

    def get_random_halo(self, npart):
        """
        Generate a random halo, with the given number of particles.
        If npart is a list, a random element of the list is chosen.

        To get a rough idea of the ranges found in a typical halo, this is the
        input for one halo from a test run on a 400 Mpc FLAMINGO box:
        (input_type, units, min value, max value)
        types = [
        "PartType0": {
          "Coordinates": (np.float64, a*snap_length, 0., boxsize),
          "GroupNr_bound": (np.int32, dimensionless, N/A),
          "LastAGNFeedbackScaleFactors": (np.float32, dimensionless, 0., 1.),
          "Masses": (np.float32, snap_mass, 0.1, 0.1),
          "MetalMassFractions": (np.float32, dimensionless, 0., 0.06),
          "SmoothedElementMassFractions":
            (np.float32, dimensionless,
             [0.68, 0.24, 0., 0., 0., 0., 0., 0., 0.],
             [0.75, 0.29, 0.006, 0.001, 0.01, 0.002, 0.0008, 0.002, 0.002]),
          "StarFormationRates": (np.float32, snap_mass/snap_time, -0.99, 246.5),
          "Temperatures": (np.float32, snap_temperature, 1.e3, 1.e10),
          "Velocities": (np.float32, snap_length/snap_time, -1.e3, 1.e3),
        },
        "PartType1": {
          "Coordinates": (np.float64, a*snap_length, 0., boxsize),
          "GroupNr_bound": (np.int32, dimensionless, N/A),
          "Masses": (np.float32, snap_mass, 0.5, 0.5),
          "Velocities": (np.float32, snap_length/snap_time, -1.e3, 1.e3),
        }
        "PartType4": {
          "Coordinates": (np.float64, a*snap_length, 0., boxsize),
          "GroupNr_bound": (np.int32, dimensionless, N/A),
          "InitialMasses": (np.float32, snap_mass, 0.1, 0.3),
          "Luminosities": (np.float32, dimensionless, 1.e5, 1.e10),
          "Masses": (np.float32, snap_mass, 0.06, 0.1),
          "MetalMassFractions": (np.float32, dimensionless, 0., 0.075),
          "Velocities": (np.float32, snap_length/snap_time, -1.e3, 1.e3),
        }
        "PartType5": {
          "AccretionRates": (np.float32, snap_mass/snap_time, 0., 0.07),
          "Coordinates": (np.float64, a*snap_length, 0., boxsize),
          "DynamicalMasses": (np.float32, snap_mass, 0.1, 0.1),
          "GroupNr_bound": (np.int32, dimensionless, N/A),
          "LastAGNFeedbackScaleFactors": (np.float32, dimensionless, 0., 1.),
          "ParticleIDs": (np.int64, dimensionless, N/A),
          "SubgridMasses": (np.float32, snap_mass, 0.00001, 0.1),
          "Velocities": (np.float32, snap_length/snap_time, -1.e3, 1.e3),
        }
        """

        if isinstance(npart, list):
            npart = np.random.choice(npart)

        reg = self.unit_registry

        centre = unyt.unyt_array(
            100.0 * np.random.random(3),
            dtype=np.float64,
            units="snap_length",
            registry=reg,
        )
        # the random halo always gets GroupNr 1
        groupnr_halo = 1

        # Generate a random radius from an exponential distribution.
        # The chosen beta parameter should ensure that ~90% of the values is
        # below 50 kpc.
        radius = np.random.exponential(1.0 / 60.0, npart)
        # generate a random direction to convert the radius into an actual
        # coordinate
        phi = 2.0 * np.pi * np.random.random(npart)
        sintheta = 2.0 * np.random.random(npart) - 1.0
        costheta = np.sqrt((1.0 - sintheta) * (1.0 + sintheta))
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        coords = np.zeros((npart, 3))
        coords[:, 0] = radius * cosphi * sintheta
        coords[:, 1] = radius * sinphi * sintheta
        coords[:, 2] = radius * costheta
        coords = unyt.unyt_array(
            coords, dtype=np.float64, units="snap_length", registry=reg
        )
        # add the (random) halo centre
        coords += centre
        mass = unyt.unyt_array(
            0.1 + 0.4 * np.random.random(npart),
            dtype=np.float32,
            units="snap_mass",
            registry=reg,
        )
        vs = unyt.unyt_array(
            1000.0 * (np.random.random((npart, 3)) - 0.5),
            dtype=np.float32,
            units="snap_length/snap_time",
            registry=reg,
        )

        # randomly allocate particle types
        # we expect halos to be dominated by DM and star particles and have
        # relatively little BH particles
        types = np.random.choice(
            ["PartType0", "PartType1", "PartType4", "PartType5"],
            size=npart,
            p=[0.2, 0.4, 0.39, 0.01],
        )
        # randomly assign bound particles to the halo
        # we make sure most particles will be bound, and use two different
        # alternative values just in case that matters
        groupnr = unyt.unyt_array(
            np.random.choice([groupnr_halo, 2, 3], size=npart, p=[0.6, 0.2, 0.2]),
            dtype=np.int32,
            units=unyt.dimensionless,
            registry=reg,
        )

        data = {}
        # gas particle variables
        gas_mask = types == "PartType0"
        Ngas = int(gas_mask.sum())
        if Ngas > 0:
            data["PartType0"] = {}
            data["PartType0"]["Coordinates"] = coords[gas_mask]
            data["PartType0"]["GroupNr_bound"] = groupnr[gas_mask]
            # we assume a fixed "snapshot" redshift of 0.1, so we make sure
            # the random values span a range of scale factors that is lower
            data["PartType0"]["LastAGNFeedbackScaleFactors"] = unyt.unyt_array(
                1.0 / 1.1 + 0.01 * np.random.random(Ngas),
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            data["PartType0"]["Masses"] = mass[gas_mask]
            data["PartType0"]["MetalMassFractions"] = unyt.unyt_array(
                1.0e-2 * np.random.random(Ngas),
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            # all entries in the element mass fractions have their own limits,
            # so we need to generate those separately if we want a realistic
            # sample
            semf = np.zeros((Ngas, 9))
            semf[:, 0] = 0.68 + 0.07 * np.random.random(Ngas)
            semf[:, 1] = 0.25 + 0.04 * np.random.random(Ngas)
            semf[:, 2] = 0.006 * np.random.random(Ngas)
            semf[:, 3] = 0.001 * np.random.random(Ngas)
            semf[:, 4] = 0.01 * np.random.random(Ngas)
            semf[:, 5] = 0.002 * np.random.random(Ngas)
            semf[:, 6] = 0.001 * np.random.random(Ngas)
            semf[:, 7] = 0.002 * np.random.random(Ngas)
            semf[:, 8] = 0.002 * np.random.random(Ngas)
            data["PartType0"]["SmoothedElementMassFractions"] = unyt.unyt_array(
                semf,
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            # negative StarFormationRates are actually scale factors, again
            # limited by the highest value at z=0.1
            data["PartType0"]["StarFormationRates"] = unyt.unyt_array(
                300.0 * np.random.random(Ngas) - 1.0 / 1.1,
                dtype=np.float32,
                units="snap_mass/snap_time",
                registry=reg,
            )
            data["PartType0"]["Temperatures"] = unyt.unyt_array(
                10.0 ** (10.0 * np.random.random(Ngas)),
                dtype=np.float32,
                units="snap_temperature",
                registry=reg,
            )
            data["PartType0"]["Velocities"] = vs[gas_mask]

        # DM properties
        dm_mask = types == "PartType1"
        Ndm = int(dm_mask.sum())
        if Ndm > 0:
            data["PartType1"] = {}
            data["PartType1"]["Coordinates"] = coords[dm_mask]
            data["PartType1"]["GroupNr_bound"] = groupnr[dm_mask]
            data["PartType1"]["Masses"] = mass[dm_mask]
            data["PartType1"]["Velocities"] = vs[dm_mask]

        # star properties
        star_mask = types == "PartType4"
        Nstar = int(star_mask.sum())
        if Nstar > 0:
            data["PartType4"] = {}
            data["PartType4"]["Coordinates"] = coords[star_mask]
            data["PartType4"]["GroupNr_bound"] = groupnr[star_mask]
            # initial masses are always larger than the actual mass
            data["PartType4"]["InitialMasses"] = unyt.unyt_array(
                mass[star_mask].value * (1.0 + np.random.random(Nstar)),
                dtype=np.float32,
                units="snap_mass",
                registry=reg,
            )
            data["PartType4"]["Luminosities"] = unyt.unyt_array(
                1.0e10 * np.random.random((Nstar, 9)),
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            data["PartType4"]["Masses"] = mass[star_mask]
            data["PartType4"]["MetalMassFractions"] = unyt.unyt_array(
                1.0e-2 * np.random.random(Nstar),
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            data["PartType4"]["Velocities"] = vs[star_mask]

        # BH properties
        bh_mask = types == "PartType5"
        Nbh = int(bh_mask.sum())
        if Nbh > 0:
            data["PartType5"] = {}
            data["PartType5"]["AccretionRates"] = unyt.unyt_array(
                0.1 * np.random.random(Nbh),
                dtype=np.float32,
                units="snap_mass/snap_time",
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
            # no need to do anything random for the IDs; we simply make sure
            # the IDs are non zero
            data["PartType5"]["ParticleIDs"] = unyt.unyt_array(
                np.arange(Nbh, dtype=np.uint64) + 1,
                dtype=np.int64,
                units=unyt.dimensionless,
                registry=reg,
            )
            # the sub-grid mass is always less than the dynamical mass
            data["PartType5"]["SubgridMasses"] = unyt.unyt_array(
                mass[bh_mask].value * (1.0e-5 + 0.99 * np.random.random(Nbh)),
                dtype=np.float32,
                units="snap_mass",
                registry=reg,
            )
            data["PartType5"]["Velocities"] = vs[bh_mask]

        # set the required halo properties
        input_halo = {}
        input_halo["cofp"] = centre
        input_halo["index"] = groupnr_halo

        return input_halo, data
