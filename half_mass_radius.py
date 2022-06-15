import numpy as np
import unyt


def get_half_mass_radius(radius, mass, total_mass):
    if total_mass == 0.0 * total_mass.units or len(mass) < 1:
        return 0.0 * radius.units

    target_mass = 0.5 * total_mass

    isort = np.argsort(radius)
    sorted_radius = radius[isort]
    # compute sum in double precision to avoid numerical overflow due to
    # weird unit conversions in unyt
    cumulative_mass = mass[isort].cumsum(dtype=np.float64)

    # np.sum() and np.cumsum() use different orders, so we have to allow for
    # some small difference
    if cumulative_mass[-1] < 0.999 * total_mass:
        raise RuntimeError(
            "Masses sum up to less than the given total mass:"
            f" cumulative_mass[-1] = {cumulative_mass[-1]},"
            f" total_mass = {total_mass}!"
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

    # we cannot use '>=', since equality would happen if half_mass_radius==0
    if half_mass_radius > sorted_radius[-1]:
        raise RuntimeError(
            "Half mass radius larger than input radii:"
            f" half_mass_radius = {half_mass_radius},"
            f" sorted_radius[-1] = {sorted_radius[-1]}!"
            f" ihalf = {ihalf}, Npart = {len(radius)},"
            f" target_mass = {target_mass},"
            f" rmin = {rmin}, rmax = {rmax},"
            f" Mmin = {Mmin}, Mmax = {Mmax},"
            f" sorted_radius = {sorted_radius},"
            f" cumulative_mass = {cumulative_mass}"
        )

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
