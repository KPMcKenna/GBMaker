import numpy as np
from itertools import product
from pymatgen.core import Structure, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.typing import ArrayLike, SpeciesLike
from typing import Dict, Sequence, List, Any
from pymatgen.analysis.structure_matcher import StructureMatcher


class GrainGenerator(SlabGenerator):
    """Class for generating Grains.

    TODO:
      * Re-write the __init__ function to make use of primitive cells whilst still
        retaining the conventional cell's miller indexing.
      * Tidy up implementation: slab -> grain, np.ndarray -> ArrayLike, etc
      * Add more comments to explain ideas behind methods.
      * Docstring the functions
    """

    def __init__(
        self,
        bulk_cell: Structure,
        miller_index: Sequence[int],
    ):
        sg = SpacegroupAnalyzer(bulk_cell)
        # primitive_cell = sg.get_primitive_standard_structure()
        # T = sg.get_conventional_to_primitive_transformation_matrix()
        unit_cell = sg.get_conventional_standard_structure()
        # primative_miller_index = np.dot(T, miller_index)
        # super().__init__(primitive_cell, primative_miller_index, None, None)
        super().__init__(unit_cell, miller_index, None, None)
        self.parent = unit_cell
        self.oriented_unit_cell.translate_sites(
            indices=range(len(self.oriented_unit_cell)),
            vector=-self.oriented_unit_cell.frac_coords.min(axis=0),
        )
        self.miller_index = miller_index

    def get_grain(self, shift: float = 0.0, tol: float = 0.1) -> "Grain":
        oriented_unit_cell = self.oriented_unit_cell.copy()
        oriented_unit_cell.translate_sites(
            indices=range(len(oriented_unit_cell)),
            vector=[0, 0, shift],
        )
        grain = Grain.from_oriented_unit_cell(
            oriented_unit_cell,
            self.miller_index,
            hkl_spacing=self.parent.lattice.d_hkl(self.miller_index),
        )
        grain.get_primitive_structure(tol)
        return grain

    def get_grains(
        self,
        bonds: Dict[Sequence[SpeciesLike], float] = None,
        ftol: float = 0.1,
        tol: float = 0.1,
        max_broken_bonds: int = 0,
        symmetrize: bool = False,
        repair: bool = False,
    ):
        c_ranges = set() if bonds is None else self._get_c_ranges(bonds)

        grains = []
        for shift in self._calculate_possible_shifts(tol=ftol):
            bonds_broken = 0
            for r in c_ranges:
                if r[0] <= shift <= r[1]:
                    bonds_broken += 1
            grain = self.get_grain(shift, tol=tol)
            if bonds_broken <= max_broken_bonds:
                grains.append(grain)
            elif repair:
                # If the number of broken bonds is exceeded,
                # we repair the broken bonds on the slab
                grains.append(self.repair_broken_bonds(grain, bonds))

        # Further filters out any surfaces made that might be the same
        m = StructureMatcher(ltol=tol, stol=tol, primitive_cell=False, scale=False)

        new_grains = []
        for g in m.group_structures(grains):
            # For each unique termination, symmetrize the
            # surfaces by removing sites from the bottom.
            if symmetrize:
                grain = g[0].symmetrize_surfaces()
                new_grains.extend(grain)
            else:
                new_grains.append(g[0])

        match = StructureMatcher(ltol=tol, stol=tol, primitive_cell=False, scale=False)
        new_grains = [g[0] for g in match.group_structures(new_grains)]
        return new_grains


class Grain(Structure):
    """A grain class the builds upon a pymatgen Structure.

    TODO:
      * Add x & y mirroring. (a & b?)
      * Rotate grain and oriented unit cell a -> x for neatness.
      * Add missing docstrings.
      * Add a pretty print for the structure.
    """

    def __init__(
        self,
        oriented_unit_cell: Structure,
        miller_index: Sequence[float],
        grain: Structure,
        mirror_x: bool = False,
        mirror_y: bool = False,
        mirror_z: bool = False,
        bulk_repeats: int = 1,
        hkl_spacing: float = None,
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
        **kwargs: The keyword arguments required to make a Structure describing
                  the minimum amount of atoms that can represent the grains
                  surfaces.
        """
        super().__init__(
            grain.lattice,
            grain.species,
            grain.frac_coords.tolist(),
            grain.charge,
            site_properties=grain.site_properties,
        )
        self.miller_index = miller_index
        self.hkl_spacing = hkl_spacing
        self.oriented_unit_cell = oriented_unit_cell
        R = rotation(np.cross(*oriented_unit_cell.lattice.matrix[:2]))
        symmop = SymmOp.from_rotation_and_translation(rotation_matrix=R)
        self.oriented_unit_cell.apply_operation(symmop)
        self.bulk_thickness = self.oriented_unit_cell.lattice.matrix[2, 2]
        self._thickness_n = bulk_repeats
        R = rotation(np.cross(*self.lattice.matrix[:2]))
        symmop = SymmOp.from_rotation_and_translation(rotation_matrix=R)
        self.apply_operation(symmop)
        self.oriented_unit_cell.sort(key=lambda site: site.frac_coords[2])
        self.mirror_x = mirror_x
        self.mirror_y = mirror_y
        self.mirror_z = mirror_z

    @property
    def thickness(self) -> float:
        """The thickness of the grain in Angstrom."""
        self._thickness = self.cart_coords.max(axis=0)[2]
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
        self.bulk_repeats += int(n)

    @thickness.setter
    def hkl_thickness(self, thickness: float):
        """Sets the thickness of the grain as close as possible to the supplied value.

        This method will set the thickness of the grain to the value supplied,
        or as close as possible. It does this by calculating the required amount
        of oriented unit cells to add to the surface representation of the grain.
        If the value supplied is not a multiple of the bulk thickness then the
        smallest amount of bulk repeats are added to make the thickness larger
        than the supplied value.

        thickness: The desired thickness of the grain in Angstrom.
        """
        # if hkl_spacing wasn't supplied try and figure it out.
        if self.hkl_spacing is None:
            self.hkl_spacing = (
                SpacegroupAnalyzer(self.oriented_unit_cell)
                .get_conventional_standard_structure()
                .lattice.d_hkl(self.miller_index)
            )
        thickness *= self.hkl_spacing
        thickness -= self.thickness
        n = np.ceil(thickness / self.bulk_thickness)
        self.bulk_repeats += int(n)

    @property
    def bulk_repeats(self) -> int:
        """The amount of bulk repeats required to achieve the desired thickness."""
        return self.__getattribute__("_thickness_n")

    @bulk_repeats.setter
    def bulk_repeats(self, n: int):
        """Sets the amount of repeats of the oriented unit cell in the grain.

        This method sets the number of oriented unit cells to be present in the
        exported grain.

        n: How many repeats of the oriented unit cell to include.
        """
        # convert n to be how many more/less unit cells are required.
        n -= self.bulk_repeats
        thickness = self.scaled_c_vector
        self._thickness += n * self.oriented_unit_cell.lattice.matrix[2, 2]
        self._thickness_n += n
        if n < 0:
            # if the sites are sorted by c then the first |n| lots of bulk cell
            # atoms need to be removed.
            self.sort(key=lambda site: site.frac_coords[2], reverse=True)
            self.remove_sites(range(-n * len(self.oriented_unit_cell)))
            self.c_vector += n * self.oriented_unit_cell.lattice.matrix[2]
        elif n > 0:
            bulk_c_vector = self.oriented_unit_cell.lattice.matrix[2]
            mirror_array = np.power([-1, -1, -1], self.mirror_array)
            self.c_vector += n * bulk_c_vector
            mirror_vector = thickness + bulk_c_vector
            mirror_vector *= float(self.mirror_z) * -mirror_array
            if not self.mirror_z:
                self.translate_sites(
                    indices=range(len(self)),
                    vector=n * bulk_c_vector,
                    frac_coords=False,
                    to_unit_cell=False,
                )
            for _n in reversed(range(n)):
                for site in reversed(self.oriented_unit_cell.sites):
                    coords = site.coords * mirror_array
                    coords += _n * bulk_c_vector + mirror_vector
                    self.insert(
                        0,
                        site.species,
                        coords,
                        coords_are_cartesian=True,
                        properties=site.properties,
                    )

    @property
    def c_vector(self) -> np.ndarray:
        return self.lattice.matrix[2].copy()

    @c_vector.setter
    def c_vector(self, c: np.ndarray):
        lattice = self.lattice.matrix.copy()
        lattice[2] = c
        lattice = Lattice(lattice)
        c_ratio = lattice.c / self.lattice.c
        self._lattice = lattice
        for site in self._sites:
            site.frac_coords /= [1, 1, c_ratio]
            site.lattice = lattice

    @property
    def scaled_c_vector(self) -> float:
        """The thickness of the grain in Angstrom."""
        k = self.c_vector / self.lattice.c
        return (self.thickness / k[2]) * k

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
            thickness = self.thickness / self.lattice.matrix[2, 2]
            for site in self._sites:
                coords = [site.coords[0], site.coords[1], -site.coords[2]]
                coords += thickness * self.c_vector
                site.coords = coords

    def as_dict(self, verbosity: int = 1, fmt: str = None, **kwargs):
        attr = ["oriented_unit_cell", "mirror_x", "mirror_y", "mirror_z", "hkl_spacing"]
        d = {k: v for k, v in self.__dict__.items() if k in attr}
        d.update(super().as_dict(verbosity=verbosity, fmt=fmt, **kwargs))
        return d

    def copy(self, site_properties: Dict[str, List[Any]] = None) -> "Grain":
        properties = self.site_properties
        if site_properties is not None:
            properties.update(site_properties)
        return Grain(
            self.oriented_unit_cell,
            self.miller_index,
            self,
            mirror_x=self.mirror_x,
            mirror_y=self.mirror_y,
            mirror_z=self.mirror_z,
            bulk_repeats=self.bulk_repeats,
            hkl_spacing=self.hkl_spacing,
        )

    def symmetrize_surfaces(self, tol: float = 1e-3) -> Sequence["Grain"]:
        """Attempt to symmetrize both surfaces of the grain.

        This method is a reworking of the pymatgen method for SlabGenerator:
        [nonstoichiometric_symmetrized_slab](https://github.com/materialsproject/pymatgen/blob/v2022.0.4/pymatgen/core/surface.py#L1308-L1361).
        Much of the code is lifted from that with comments where it has been
        changed.
        """
        sg = SpacegroupAnalyzer(self, symprec=tol)
        if sg.is_laue():
            return [self.copy()]
        nonstoich_slabs = []
        for top in [True, False]:
            slab = self.copy()
            # set the bulk thickness to 2 this means that
            slab.bulk_repeats = 2
            slab.sort(key=lambda site: site.frac_coords[2], reverse=not top)
            # maybe add energy at some point
            # slab.energy = init_slab.energy

            for _ in self.oriented_unit_cell.sites:
                # Keep removing sites from the TOP one by one until both
                # surfaces are symmetric or the number of sites IS EQUAL TO THE
                # NUMBER OF ATOMS IN THE ORIENTED UNIT CELL.
                slab.remove_sites([len(slab) - 1])
                # Check if the altered surface is symmetric
                sg = SpacegroupAnalyzer(slab, symprec=tol)
                if sg.is_laue():
                    if not top:
                        vector = [0, 0, -slab.frac_coords.min(axis=0)[2]]
                        vector = slab.lattice.get_cartesian_coords(vector)
                        slab.translate_sites(
                            indices=range(len(slab)),
                            vector=vector,
                            frac_coords=False,
                        )
                        slab.oriented_unit_cell.translate_sites(
                            indices=range(len(slab.oriented_unit_cell)),
                            vector=vector,
                            frac_coords=False,
                        )
                    # reset the slab thickness as we have removed atoms,
                    # reducing the bulk thickness.
                    slab._thickness_n = 1
                    nonstoich_slabs.append(slab)
                    break
        return nonstoich_slabs

    @classmethod
    def from_oriented_unit_cell(
        cls,
        oriented_unit_cell: Structure,
        miller_index: Sequence[float],
        mirror_x: bool = False,
        mirror_y: bool = False,
        mirror_z: bool = False,
        bulk_repeats: int = 1,
        hkl_spacing: float = None,
    ) -> "Grain":
        grain = cls(
            oriented_unit_cell,
            miller_index,
            oriented_unit_cell,
            mirror_x,
            mirror_y,
            mirror_z,
            bulk_repeats,
            hkl_spacing,
        )
        c_vector = grain.c_vector
        c_vector /= grain.oriented_unit_cell.lattice.c
        c_vector *= 5.0 / c_vector[2]
        grain.c_vector += c_vector
        return grain


class GrainBoundary:
    """A grain boundary for storing grains.

    TODO:
      * Docstrings.
      * Lattice matching -> scale 2nd grain (find Moire lattice?).
    """

    def __init__(
        self,
        grain_1: Grain,
        grain_2: Grain = None,
        mirror_x: bool = False,
        mirror_y: bool = False,
        mirror_z: bool = False,
        translation_vec: Sequence[float] = [0.0, 0.0, 0.0],
        vacuum: float = None,
        merge_tol: float = None,
    ):
        self.grain_1 = grain_1
        self.grain_2 = grain_1.copy() if grain_2 is None else grain_2
        self.grain_2.mirror_x = mirror_x
        self.grain_2.mirror_y = mirror_y
        self.grain_2.mirror_z = mirror_z
        self.translation_vec = np.array(translation_vec)
        self.vacuum = translation_vec[2] if vacuum is None else vacuum
        self.merge_tol = merge_tol

    def translate(
        self,
        vector: ArrayLike,
        frac_coords: bool = False,
        z_on_both_interfaces: bool = True,
    ):
        if frac_coords:
            vector += self.grain_1.lattice.get_cartesian_coords(vector)
        self.translation_vec += vector
        if z_on_both_interfaces:
            self.vacuum += vector[2]

    def scan(
        self,
        na: int,
        nb: int,
        nc: int = 1,
        da: float = None,
        db: float = None,
        dc: float = None,
        z_on_both_interfaces: bool = True,
    ):
        """An iterator that yeilds grainboundary Structures for scanning over the grains.

        hello.
        """
        da = 1 / na if da is None else da
        db = 1 / nb if db is None else db
        dc = 1 / nc if dc is None else dc
        a = self.grain_1.lattice.matrix[0] * da
        b = self.grain_1.lattice.matrix[1] * db
        c = self.grain_1.lattice.matrix[2] * dc
        tv = self.translation_vec
        vacuum = self.vacuum
        for ia, ib, ic in product(range(na), range(nb), range(nc)):
            translation_vec = tv + ia * a
            translation_vec += ib * b
            translation_vec += ic * c
            self.translation_vec = translation_vec
            if z_on_both_interfaces:
                self.vacuum = vacuum + ic * dc
            yield self.as_structure()

    def convergence(
        self,
        bulk_repeats_1: Sequence[int],
        bulk_repeats_2: Sequence[int] = None,
    ):
        bulk_repeats_2 = bulk_repeats_1 if bulk_repeats_2 is None else bulk_repeats_2
        for n_1, n_2 in zip(bulk_repeats_1, bulk_repeats_2):
            if n_1 == 0 or n_2 == 0:
                continue
            self.grain_1.bulk_repeats = n_1
            self.grain_2.bulk_repeats = n_2
            yield self.as_structure()

    def as_structure(self) -> Structure:
        height = (
            self.grain_1.thickness
            + self.grain_2.thickness
            + self.translation_vec[2]
            + self.vacuum
        )
        lattice = self.grain_1.lattice.matrix.copy()
        lattice[2] /= self.grain_1.lattice.c
        lattice[2] *= height / lattice[2, 2]
        translation_vec = self.translation_vec + self.grain_1.scaled_c_vector
        coords = np.add(self.grain_2.cart_coords, translation_vec)
        site_properties = self.grain_1.site_properties
        for key, value in site_properties.items():
            value_2 = self.grain_2.site_properties.get(key)
            if value_2 is None:
                site_properties.pop(key)
            else:
                value.extend(value_2)
                site_properties[key] = value
        site_properties["grain"] = np.concatenate(
            [np.repeat(0, len(self.grain_1)), np.repeat(1, len(self.grain_2))]
        )
        grain_boundary = Structure(
            lattice,
            np.concatenate([self.grain_1.species, self.grain_2.species]),
            np.concatenate([self.grain_1.cart_coords, coords]),
            to_unit_cell=True,
            coords_are_cartesian=True,
            site_properties=site_properties,
        )
        if self.merge_tol is not None:
            grain_boundary.merge_sites(tol=self.merge_tol, mode="delete")
        return grain_boundary


def rotation(vector: ArrayLike, axis: ArrayLike = [0, 0, 1]) -> np.ndarray:
    r = np.array(vector) / np.linalg.norm(vector)
    theta, b = np.arccos(r[2]), np.cross(r, np.array(axis))
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
