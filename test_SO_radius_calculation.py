#!/bin/env python

import numpy as np
import unyt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl

from SO_properties import find_SO_radius_and_mass
from halo_properties import ReadRadiusTooSmallError


def test_SO_radius_calculation():

    npart = 10
    Mpart = 1.0e9 * unyt.Msun

    np.random.seed(62)

    radius = np.random.exponential(1.0, npart) * unyt.kpc
    rmax = radius.max()

    ordered_radius = np.sort(radius)

    mass = Mpart * (1.0 + 0.2 * (np.random.random(npart) - 0.5))
    cumulative_mass = np.cumsum(mass)

    ipos = np.argmax(ordered_radius > 0.0)
    ordered_radius = ordered_radius[ipos:]
    cumulative_mass = cumulative_mass[ipos:]
    density = cumulative_mass / (4.0 * np.pi / 3.0 * ordered_radius**3)

    reference_density = 200.0 * Mpart * npart / (4.0 * np.pi / 3.0 * rmax**3)

    try:
        SO_r, SO_mass = find_SO_radius_and_mass(
            ordered_radius, density, cumulative_mass, reference_density
        )
        print(f"SO_r: {SO_r}, SO_mass: {SO_mass}")
    except ReadRadiusTooSmallError:
        print("Radius too small!")

    fig, ax = pl.subplots(2, 2, sharex="col")

    ordered_radius.convert_to_units("kpc")
    density.convert_to_units("g/cm**3")
    cumulative_mass.convert_to_units("Msun")
    reference_density.convert_to_units("g/cm**3")
    SO_r.convert_to_units("kpc")
    SO_mass.convert_to_units("Msun")

    ax[0][0].semilogy(ordered_radius, density, "o-")
    ax[1][0].semilogy(ordered_radius, cumulative_mass, "o-")
    rrange = np.linspace(0.0, 1.1 * SO_r, 100)
    Mrange = reference_density * 4.0 * np.pi / 3.0 * rrange**3
    rrange.convert_to_units("kpc")
    Mrange.convert_to_units("Msun")
    ax[1][0].semilogy(rrange, Mrange, ":", color="C2")
    icross = np.argmin(np.abs(ordered_radius - SO_r))
    beg = max(0, icross - 10)
    end = min(len(ordered_radius) - 1, icross + 10)
    ax[0][1].semilogy(ordered_radius[beg:end], density[beg:end], "o-")
    ax[1][1].semilogy(ordered_radius[beg:end], cumulative_mass[beg:end], "o-")
    rrange = np.linspace(0.9 * SO_r, 1.1 * SO_r, 100)
    Mrange = reference_density * 4.0 * np.pi / 3.0 * rrange**3
    rrange.convert_to_units("kpc")
    Mrange.convert_to_units("Msun")
    ax[0][1].semilogy(
        rrange,
        (cumulative_mass[icross] / (4.0 * np.pi / 3.0 * rrange**3)).to("g/cm**3"),
        ":",
        color="C2",
    )
    ax[1][1].semilogy(rrange, Mrange, ":", color="C2")

    ax[0][0].axhline(y=reference_density, linestyle="--", color="C1")
    ax[1][0].axhline(y=SO_mass, linestyle="--", color="C1")
    ax[0][0].axvline(x=SO_r, linestyle="--", color="C1")
    ax[1][0].axvline(x=SO_r, linestyle="--", color="C1")
    ax[0][1].axhline(y=reference_density, linestyle="--", color="C1")
    ax[1][1].axhline(y=SO_mass, linestyle="--", color="C1")
    ax[0][1].axvline(x=SO_r, linestyle="--", color="C1")
    ax[1][1].axvline(x=SO_r, linestyle="--", color="C1")
    ax[0][1].plot(SO_r, reference_density, "kx")
    ax[1][1].plot(SO_r, SO_mass, "kx")

    ax[0][0].set_ylabel("density")
    ax[1][0].set_ylabel("cumulative mass")
    ax[1][0].set_xlabel("radius")
    ax[1][1].set_xlabel("radius")

    pl.tight_layout()
    pl.savefig("test_SO_radius.png", dpi=300)
    fig.clear()
    pl.close()


if __name__ == "__main__":
    test_SO_radius_calculation()
