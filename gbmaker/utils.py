import numpy as np
from numpy.typing import ArrayLike
from pymatgen.core import Structure


def orthogonalise(s: Structure) -> Structure:
    """Attempt to orthogonalise the structure.

    Using the fact that a lies on the x axis and b lies in the x-y plane attempt
    to orthogonalise the lattice by either adding or subtracting integer multiples
    a from b and then b from c and a from the modified c.
    """
    l = s.lattice.matrix.copy()
    a1 = l[0, 0]
    b1, b2 = l[1, 0:2]
    c1, c2, c3 = l[2]
    b1 += round(-b1 / a1) * a1
    n = round(-c2 / b2)
    c1 += n * b1
    c2 += n * b2
    c1 += round(-c1 / a1) * a1
    lattice = [[a1, 0, 0], [b1, b2, 0], [c1, c2, c3]]
    return Structure(
        lattice,
        s.species,
        s.cart_coords.tolist(),
        s.charge,
        to_unit_cell=True,
        coords_are_cartesian=True,
        site_properties=s.site_properties,
    )


def rotation(vector: ArrayLike, axis: ArrayLike = [0, 0, 1]) -> np.ndarray:
    """Calculate the rotation matrix for aligning two vectors."""
    r = np.array(vector) / np.linalg.norm(vector)
    theta, b = np.arccos(np.dot(r, axis)), np.cross(r, np.array(axis))
    mag_b = np.linalg.norm(b)
    if mag_b == 0:
        return np.eye(3)
    b = b / mag_b
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    q = [c, *np.multiply(s, b)]
    Q = np.array(
        [
            [
                q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2,
                2 * (q[1] * q[2] - q[0] * q[3]),
                2 * (q[1] * q[3] + q[0] * q[2]),
            ],
            [
                2 * (q[1] * q[2] + q[0] * q[3]),
                q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2,
                2 * (q[2] * q[3] - q[0] * q[1]),
            ],
            [
                2 * (q[1] * q[3] - q[0] * q[2]),
                2 * (q[2] * q[3] + q[0] * q[1]),
                q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2,
            ],
        ]
    )
    return Q


def float_gcd(a, b, rtol=1e-05, atol=1e-08):
    """Compute the gcd of two floats.

    https://stackoverflow.com/a/45325587
    """
    t = min(abs(a), abs(b))
    while abs(b) > rtol * t + atol:
        a, b = b, a % b
    return a
