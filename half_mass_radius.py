import numpy as np
import unyt


def get_half_mass_radius(radius, mass, total_mass):
    if total_mass == 0.0 * total_mass.units or len(mass) < 1:
        return 0.0 * radius.units

    target_mass = 0.5 * total_mass

    isort = np.argsort(radius)
    sorted_radius = radius[isort]
    cumulative_mass = mass[isort].cumsum()

    # np.sum() and np.cumsum() use different orders, so we have to allow for
    # some small difference
    if cumulative_mass[-1] < 0.999 * total_mass:
        raise RuntimeError(
            f"Masses sum up to less than the given total mass: cumulative_mass[-1] = {cumulative_mass[-1]}, total_mass = {total_mass}!"
        )

    ihalf = np.argmax(cumulative_mass >= target_mass)
    if ihalf == 0:
        rmin = 0.0 * radius.units
        Mmin = 0.0 * mass.units
    else:
        rmin = sorted_radius[ihalf - 1]
        Mmin = cumulative_mass[ihalf - 1]
    rmax = sorted_radius[ihalf]
    Mmax = cumulative_mass[ihalf]

    if Mmin == Mmax:
        half_mass_radius = 0.5 * (rmin + rmax)
    else:
        half_mass_radius = rmin + (target_mass - Mmin) / (Mmax - Mmin) * (rmax - rmin)

    return half_mass_radius


def test_get_half_mass_radius():
    np.random.seed(203)

    for i in range(1000):
        npart = np.random.choice([1, 10, 100, 1000, 10000])

        radius = np.random.exponential(1.0, npart) * unyt.kpc

        Mpart = 1.0e9 * unyt.Msun
        mass = Mpart * (1.0 + 0.2 * (np.random.random(npart) - 0.5))

        total_mass = mass.sum()

        half_mass_radius = get_half_mass_radius(radius, mass, total_mass)

        mask = radius <= half_mass_radius
        Mtest = mass[mask].sum()
        assert Mtest <= 0.5 * total_mass

    fail = False
    try:
        half_mass_radius = get_half_mass_radius(radius, mass, 2.0 * total_mass)
    except RuntimeError:
        fail = True
    assert fail
