from pymatgen.core import Structure, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.core.operations import SymmOp
from pymatgen.util.typing import ArrayLike
import numpy as np


class GrainGenerator(SlabGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Grain(Structure):
    """A grain class the builds upon a pymatgen Structure."""

    def __init__(
        self,
        oriented_unit_cell: Structure,
        mirror_x: bool = False,
        mirror_y: bool = False,
        mirror_z: bool = False,
        bulk_repeats: int = 1,
        **kwargs
    ):
        """Initialise the grain from an oriented unit cell and Structure kwargs.

        oriented_unit_cell: A pymatgen Structure contianing the unit cell
                            oriented such that the required millier index is in
                            the ab-plane. This cell should be shifted such that
                            the c = 0 plane is identical as the grain being
                            created.
        mirror_x: Whether to mirror the grain along the x-direction.
        mirror_y: Whether to mirror the grain along the y-direction.
        mirror_z: Whether to mirror the grain along the z-direction.
        bulk_repeats: The number of bulk repeats in the grain, default=1.
        **kwargs: The keyword arguments required to make a Structure describing
                  the minimum amount of atoms that can represent the grains
                  surfaces.
        """
        super().__init__(**kwargs)
        self.oriented_unit_cell = oriented_unit_cell
        R = rotation(np.cross(*oriented_unit_cell.lattice.matrix[:2]))
        symmop = SymmOp.from_rotation_and_translation(rotation_matrix=R)
        self.oriented_bulk_cell.apply_operation(symmop)
        self.bulk_thickness = self.oriented_unit_cell.lattice.matrix[2, 2]
        self.bulk_repeats = bulk_repeats
        R = rotation(np.cross(*self.lattice.matrix[:2]))
        symmop = SymmOp.from_rotation_and_translation(rotation_matrix=R)
        self.apply_operation(symmop)
        self.mirror_x = mirror_x
        self.mirror_y = mirror_y
        self.mirror_z = mirror_z

    @property
    def thickness(self) -> float:
        """The thickness of the grain in Angstrom."""
        try:
            return self.__getattribute__("_thickness")
        except AttributeError:
            self._thickness = self.cart_coords.max(axis=0)[2]
            self._thickness_n = 1
            return self._thickness

    @thickness.setter
    def thickness(self, thickness: float):
        """Sets the thickness of the grain as close as possible to the supplied value.

        This method will set the thickness of the grain to the value supplied,
        or as close as possible. It does this by calculating the required amount
        of oriented unit cells to add to the surface representation of the grain.
        If the value supplied is not a multiple of the bulk thickness then the
        smallest amount of bulk repeats are added to make the thickness larger
        than the supplied value.

        thickness: The desired thickness of the grain in Angstrom.
        """
        thickness -= self.thickness
        n = np.ceil(thickness / self.bulk_thickness)
        self.bulk_repeats += n

    @property
    def bulk_repeats(self) -> int:
        """The amount of bulk repeats required to achieve the desired thickness."""
        try:
            return self.__getattribute__("_thickness_n")
        except AttributeError:
            _ = self.thickness
            return 1

    @bulk_repeats.setter
    def bulk_repeats(self, n: int):
        """Sets the amount of repeats of the oriented unit cell in the grain.

        This method sets the number of oriented unit cells to be present in the
        exported grain.

        n: How many repeats of the oriented unit cell to include.
        """
        # convert n to be how many more/less unit cells are required.
        n -= self.bulk_repeats
        self._thickness += n * self.oriented_bulk_cell.lattice.matrix[2, 2]
        self._thickness_n += n
        if n < 0:
            # if the sites are sorted by c then the first |n| lots of bulk cell
            # atoms need to be removed.
            self.sort(key=lambda site: site.frac_coords[2])
            self.remove_sites(range(-n * len(self.oriented_unit_cell)))
            self.translate_sites(
                indices=range(self.len()),
                vector=[0, 0, -self.frac_coords.max(axis=0)[2]],
            )
            self.c_vector += n * self.oriented_unit_cell.lattice.matrix[2]
        elif n > 0:
            bulk_c_vector = self.oriented_unit_cell.lattice.matrix[2]
            self.c_vector += n * bulk_c_vector
            self.translate_sites(
                indices=range(self.len()),
                vector=n * self.oriented_unit_cell.lattice.matrix[2],
                frac_coords=False,
                to_unit_cell=False,
            )
            for _n in reversed(range(n)):
                for site in reversed(self.oriented_unit_cell.sites):
                    self.insert(
                        0,
                        site.species,
                        site.coords + _n * bulk_c_vector,
                        coords_are_cartesian=True,
                        properties=site.properties,
                    )

    @property
    def c_vector(self) -> ArrayLike:
        return self.lattice.matrix[2]

    @c_vector.setter
    def c_vector(self, c: ArrayLike):
        lattice = self.lattice.matrix
        lattice[2] = c
        lattice = Lattice(lattice)
        c_ratio = lattice.c / self.lattice.c
        self._lattice = lattice
        for site in self._sites:
            site.frac_coords *= [1, 1, c_ratio]
            site.lattice = lattice

    @property
    def mirror_array(self) -> np.ndarray:
        """An array representing if any dirrections have been mirrored."""
        try:
            return self.__getattribute__("_mirror")
        except AttributeError:
            self._mirror = np.zeros(3, dtype=bool)
            return self._mirror

    @property
    def mirror_x(self) -> bool:
        """Has the grain been mirrored along the x-dirrection?"""
        return self.mirror_array[0]

    @mirror_x.setter
    def mirror_x(self, b: bool):
        if b != self.mirror_x:
            self._mirror[0] = b

    @property
    def mirror_y(self) -> bool:
        """Has the grain been mirrored along the y-dirrection?"""
        return self.mirror_array[1]

    @mirror_y.setter
    def mirror_y(self, b: bool):
        if b != self.mirror_y:
            self._mirror[1] = b

    @property
    def mirror_z(self) -> bool:
        """Has the grain been mirrored along the z-dirrection?"""
        return self.mirror_array[2]

    @mirror_z.setter
    def mirror_z(self, b: bool):
        if b != self.mirror_z:
            self._mirror[2] = b


class GrainBoundary:
    def __init__(
        self,
        grain_1: Grain,
        grain_2: Grain = None,
        mirror_x: bool = False,
        mirror_y: bool = False,
        mirror_z: bool = False,
        translation_vec: ArrayLike = [0.0, 0.0, 0.0],
        vacuum: float = None,
        merge_tol: float = None,
    ):
        self.grain_1 = grain_1
        self.grain_2 = grain_1.copy() if grain_2 is None else grain_2
        self.grain_2.mirror_x = mirror_x
        self.grain_2.mirror_y = mirror_y
        self.grain_2.mirror_z = mirror_z
        self.translation_vec = np.array(translation_vec)

    def translate(self, vector: ArrayLike):
        self.translation_vec += vector


def rotation(vector: ArrayLike, axis: ArrayLike = [0, 0, 1]) -> np.ndarray:
    if all([v == a for v, a in zip(vector, axis)]):
        return np.eye(3)
    r = np.array(vector) / np.linalg.norm(vector)
    theta, b = np.arccos(r[2]), np.cross(r, np.array(axis))
    b = b / np.linalg.norm(b)
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
