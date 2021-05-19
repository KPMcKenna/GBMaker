import numpy as np
from itertools import product
from pymatgen.core import Structure, Lattice, Site, PeriodicSite, IStructure
from pymatgen.core.surface import SlabGenerator, Slab
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.typing import SpeciesLike
from numpy.typing import ArrayLike
from typing import Dict, Sequence, Any, Callable, Optional, List, Iterator, Tuple
from pymatgen.analysis.structure_matcher import StructureMatcher
from functools import reduce


class GrainGenerator(SlabGenerator):
    """Class for generating Grains.

    TODO:
      * Tidy up implementation: slab -> grain, np.ndarray -> ArrayLike, etc
      * Add more comments to explain ideas behind methods.
    """

    def __init__(
        self,
        bulk_cell: Structure,
        miller_index: ArrayLike,
    ):
        """Initialise the GrainGenerator class from any bulk structure.

        The Grain is created with miller index relative to the conventional
        standard structure returned by pymatgen.symmetry.analyzer.SpacegroupAnalyzer.
        This class creates the oriented unit cell from a primitive cell so that the
        returned cell is fairly primitive. Attempts are made to orthogonalise
        the cell without increasing the number of atoms in the oriented unit cell.
        """
        sg = SpacegroupAnalyzer(bulk_cell)
        unit_cell = sg.get_conventional_standard_structure()
        primitive_cell = sg.get_primitive_standard_structure()
        T = sg.get_conventional_to_primitive_transformation_matrix()
        # calculate the miller index relative to the primitive cell
        primative_miller_index = np.dot(T, miller_index)
        primative_miller_index /= reduce(float_gcd, primative_miller_index)
        primative_miller_index = np.array(
            np.round(primative_miller_index, 1), dtype=int
        )
        super().__init__(primitive_cell, primative_miller_index, None, None)
        # use the conventional standard structure and the supplied miller index
        self.parent = unit_cell
        self.miller_index = np.array(miller_index)
        # first rotate the oriented unit cell so that the miller index is along z
        R = rotation(np.cross(*self.oriented_unit_cell.lattice.matrix[:2]))
        symmop = SymmOp.from_rotation_and_translation(rotation_matrix=R)
        self.oriented_unit_cell.apply_operation(symmop)
        # then rotate the oriented unit cell so that the a-vector lies along x
        theta = np.arccos(
            self.oriented_unit_cell.lattice.matrix[0, 0]
            / self.oriented_unit_cell.lattice.a
        )
        theta *= -1 if self.oriented_unit_cell.lattice.matrix[0, 1] > 0 else 1
        symmop = SymmOp.from_axis_angle_and_translation(
            [0, 0, 1],
            theta,
            True,
        )
        self.oriented_unit_cell.apply_operation(symmop)
        # try and orthogonalise the structure
        self.oriented_unit_cell = orthogonalise(self.oriented_unit_cell)

    def get_grain(self, shift: float = 0.0) -> "Grain":
        """From a given fractional shift in c in the oriented unit cell create a Grain.

        Despite the documentation in pymatgen sugguesting this shift is in Angstrom
        the shift is fractional (Pull request to change this?).
        """
        return Grain.from_oriented_unit_cell(
            self.oriented_unit_cell.copy(),
            self.miller_index.copy(),
            shift,
            hkl_spacing=self.parent.lattice.d_hkl(self.miller_index),
        )

    def get_grains(
        self,
        bonds: Optional[Dict[Sequence[SpeciesLike], float]] = None,
        ftol: float = 0.1,
        tol: float = 0.1,
        max_broken_bonds: int = 0,
        symmetrize: bool = False,
        repair: bool = False,
    ) -> Iterator["Grain"]:
        """Get the differing terminations of Grains for the oriented unit cell.

        This is the same method as get_slabs() but yeilding Grains instead of
        returning slabs.

        --Warning--
        This currently works as the structure matcher doesn't periodically wrap
        if this changes switch the __iter__() method of Grains to return a Slab.
        """
        c_ranges = set() if bonds is None else self._get_c_ranges(bonds)

        grains = []
        for shift in self._calculate_possible_shifts(tol=ftol):
            bonds_broken = 0
            for r in c_ranges:
                if r[0] <= shift <= r[1]:
                    bonds_broken += 1
            grain = self.get_grain(shift)
            if bonds_broken <= max_broken_bonds:
                grains.append(grain)
            elif repair:
                grain.bonds = bonds
                grains.append(grain)

        # Further filters out any surfaces made that might be the same
        m = StructureMatcher(ltol=tol, stol=tol, primitive_cell=False, scale=False)

        new_grains = []
        for g in m.group_structures(grains):
            g = self.get_grain(g[0].shift[2])
            # For each unique termination, symmetrize the
            # surfaces by removing sites from the top.
            # The repeat will be caught by the other termination.
            if symmetrize:
                grain = g.symmetrize_surfaces()
                if grain is not None:
                    new_grains.append(grain)
            else:
                new_grains.append(g)

        match = StructureMatcher(ltol=tol, stol=tol, primitive_cell=False, scale=False)
        new_grains = [g[0] for g in match.group_structures(new_grains)]
        for grain in new_grains:
            yield grain


class GrainBoundaryGenerator:
    """A class for generating GrainBoundary classes."""

    def __init__(
        self,
        bulk_0: Structure,
        miller_0: ArrayLike,
        bulk_1: Optional[Structure] = None,
        miller_1: Optional[ArrayLike] = None,
    ):
        """Initialise the generator with either one or two bulk structures.

        If a second bulk structure or second miller index is supplied then the
        second grain will be made of this different grain.

        TODO:
          - Moire lattice matching.
          - Rotation.
        """
        self.bulk_0 = bulk_0
        self.miller_0 = miller_0
        self.bulk_1 = bulk_1
        self.miller_1 = miller_1

    def get_grain_boundaries(
        self,
        mirror_x: bool = False,
        mirror_y: bool = False,
        mirror_z: bool = False,
        vacuum: Optional[float] = None,
        translation_vec: ArrayLike = [0.0, 0.0, 0.0],
        merge_tol: Optional[float] = None,
        reconstruction: Optional[Callable[["Grain", Site], bool]] = None,
        bonds: Optional[Dict[Sequence[SpeciesLike], float]] = None,
        ftol: float = 0.1,
        tol: float = 0.1,
        max_broken_bonds: int = 0,
        symmetrize: bool = False,
        repair: bool = False,
    ) -> Iterator["GrainBoundary"]:
        """Generate an iterator of GrainBoundaries over possible grain terminations.

        Using the arguements available to the GrainGenerator and GrainBoundary
        classes create an iterator over the available grains.
        """
        grains_0 = GrainGenerator(self.bulk_0, self.miller_0).get_grains(
            bonds, ftol, tol, max_broken_bonds, symmetrize, repair
        )
        if self.bulk_1 is not None or self.miller_1 is not None:
            bulk_1 = self.bulk_1 if self.bulk_1 is not None else self.bulk_0
            miller_1 = self.miller_1 if self.miller_1 is not None else self.miller_0
            grains_1 = GrainGenerator(bulk_1, miller_1).get_grains(
                bonds, ftol, tol, max_broken_bonds, symmetrize, repair
            )

            def map_func_gs(grains):
                return GrainBoundary(
                    grains[0],
                    grains[1],
                    mirror_x=mirror_x,
                    mirror_y=mirror_y,
                    mirror_z=mirror_z,
                    vacuum=vacuum,
                    translation_vec=translation_vec,
                    merge_tol=merge_tol,
                    reconstruction=reconstruction,
                )

            gb_map = map(map_func_gs, product(grains_0, grains_1))
        else:

            def map_func_g(grain):
                return GrainBoundary(
                    grain,
                    mirror_x=mirror_x,
                    mirror_y=mirror_y,
                    mirror_z=mirror_z,
                    vacuum=vacuum,
                    translation_vec=translation_vec,
                    merge_tol=merge_tol,
                    reconstruction=reconstruction,
                )

            gb_map = map(map_func_g, grains_0)
        for gb in gb_map:
            yield gb


class Grain:
    """A grain class the builds upon a pymatgen Structure.

    TODO:
      * Add missing docstrings.
      * Add a pretty print for the structure.
    """

    def __init__(
        self,
        oriented_unit_cell: Structure,
        miller_index: ArrayLike,
        base: Structure,
        shift: ArrayLike,
        mirror_x: bool = False,
        mirror_y: bool = False,
        mirror_z: bool = False,
        hkl_spacing: Optional[float] = None,
        bonds: Optional[Dict[Sequence[SpeciesLike], float]] = None,
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
        self.base = IStructure.from_sites(base)
        self.miller_index = np.array(miller_index)
        self.shift = np.array(shift)
        self.hkl_spacing = hkl_spacing
        self.oriented_unit_cell = IStructure.from_sites(oriented_unit_cell)
        try:
            self.oriented_unit_cell._sites[0].bulk_equivalent
        except AttributeError:
            sg = SpacegroupAnalyzer(self.oriented_unit_cell)
            self.oriented_unit_cell.add_site_property(
                "bulk_equivalent",
                sg.get_symmetry_dataset()["equivalent_atoms"].tolist(),
            )
        self.bulk_thickness = self.oriented_unit_cell.lattice.matrix[2, 2]
        self.bulk_repeats = 1
        self.ab_scale = [1, 1]
        self._mirror = np.zeros(3, dtype=bool)
        self.mirror_x = mirror_x
        self.mirror_y = mirror_y
        self.mirror_z = mirror_z
        self._translation_vec = np.zeros(3, dtype=float)
        self.bonds = bonds

    def __len__(self) -> int:
        ouc_len = (
            (self.bulk_repeats - 1)
            * np.product(self.ab_scale)
            * len(self.oriented_unit_cell)
        )
        base_len = np.product(self.ab_scale) * len(self.base)
        return ouc_len + base_len

    @property
    def lattice(self) -> Lattice:
        lattice = self.base.lattice.matrix.copy()
        lattice[0] *= self.ab_scale[0]
        lattice[1] *= self.ab_scale[1]
        lattice[2] /= lattice[2, 2]
        lattice[2] *= self.thickness
        return Lattice(lattice)

    @property
    def charge(self) -> Optional[float]:
        try:
            chg = (
                self.oriented_unit_cell.charge * (self.bulk_repeats - 1)
                + self.base.charge
            )
            chg *= np.product(self.ab_scale)
        except TypeError:
            chg = None
        return chg

    @property
    def sites(self) -> List[PeriodicSite]:
        return list(self)

    def __iter__(self) -> Iterator[PeriodicSite]:
        mirror = np.power([-1, -1, -1], self._mirror)
        z_shift = int(self.mirror_z) * self.lattice.matrix[2] * -1 * mirror
        try:
            ouc = self.oriented_unit_cell * [*self.ab_scale, self.bulk_repeats - 1]
            ouc.translate_sites(
                range(len(ouc)),
                self.translation_vec - self.shift,
                frac_coords=False,
                to_unit_cell=False,
            )
            ouc_c = ouc.lattice.matrix[2]
            for site in ouc.sites:
                c = np.round(site.frac_coords, 8) % 1
                c = ouc.lattice.get_cartesian_coords(c)
                c *= mirror
                c += z_shift
                c = np.round(self.lattice.get_fractional_coords(c), 8)
                yield PeriodicSite(
                    site.species,
                    c,
                    self.lattice,
                    to_unit_cell=False,
                    properties=site.properties,
                )
        except np.linalg.LinAlgError:
            ouc_c = np.zeros(3)
        base = self.base * [*self.ab_scale, 1]
        base.translate_sites(
            range(len(base)),
            self.translation_vec,
            frac_coords=False,
        )
        for site in base:
            c = np.round(site.frac_coords, 8) % 1
            c = base.lattice.get_cartesian_coords(c)
            c = c + ouc_c
            c *= mirror
            c += int(self.mirror_z) * self.lattice.matrix[2] * -1 * mirror
            c = np.round(self.lattice.get_fractional_coords(c), 8)
            yield PeriodicSite(
                site.species,
                c,
                self.lattice,
                to_unit_cell=False,
                properties=site.properties,
            )

    def __getitem__(self, ind: int) -> PeriodicSite:
        return self.sites[ind]

    @property
    def thickness(self) -> float:
        """The thickness of the grain in Angstrom."""
        b = (self.bulk_repeats - 1) * self.oriented_unit_cell.lattice.matrix[2, 2]
        return self.base.cart_coords.max(axis=0)[2] + b

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
                .lattice.d_hkl(self.miller_index.copy())
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
        self._thickness_n = int(n)

    @property
    def mirror_x(self):
        return self._mirror[0]

    @mirror_x.setter
    def mirror_x(self, b: bool):
        self._mirror[0] = b

    @property
    def mirror_y(self):
        return self._mirror[1]

    @mirror_y.setter
    def mirror_y(self, b: bool):
        self._mirror[1] = b

    @property
    def mirror_z(self):
        return self._mirror[2]

    @mirror_z.setter
    def mirror_z(self, b: bool):
        self._mirror[2] = b

    def copy(self) -> "Grain":
        grain = Grain(
            self.oriented_unit_cell.copy(),
            self.miller_index.copy(),
            self.base.copy(),
            self.shift.copy(),
            self.mirror_x,
            self.mirror_y,
            self.mirror_z,
            self.hkl_spacing,
        )
        grain.bulk_repeats = self.bulk_repeats
        return grain

    def orthogonalise_c(self):
        lattice = self.base.lattice.matrix.copy()
        lattice[2, :2] = 0
        lattice = Lattice(lattice)
        coords = self.base.cart_coords
        self.base._lattice = lattice
        for c, site in zip(coords, self.base._sites):
            site._lattice = lattice
            site.coords = c

    def symmetrize_surfaces(self, tol: float = 1e-3) -> Optional["Grain"]:
        """Attempt to symmetrize both surfaces of the grain.

        This method is a reworking of the pymatgen method for SlabGenerator:
        [nonstoichiometric_symmetrized_slab](https://github.com/materialsproject/pymatgen/blob/v2022.0.4/pymatgen/core/surface.py#L1308-L1361).
        Much of the code is lifted from that with comments where it has been
        changed.
        """
        sg = SpacegroupAnalyzer(self.base, symprec=tol)
        if sg.is_laue():
            return self.copy()
        br = self.bulk_repeats
        self.bulk_repeats = 2

        slab = self.get_slab()
        # set the bulk thickness to 2 this means that
        slab.sort(key=lambda site: (site.z, site.bulk_equivalent))
        # maybe add energy at some point
        # slab.energy = init_slab.energy
        grain = None

        for _ in self.oriented_unit_cell.sites:
            # Keep removing sites from the TOP one by one until both
            # surfaces are symmetric or the number of sites IS EQUAL TO THE
            # NUMBER OF ATOMS IN THE ORIENTED UNIT CELL.
            slab.remove_sites([len(slab) - 1])
            # Check if the altered surface is symmetric
            sg = SpacegroupAnalyzer(slab, symprec=tol)
            if sg.is_laue():
                # reset the slab thickness as we have removed atoms,
                # reducing the bulk thickness.
                slab.sort(key=lambda site: (round(site.z, 8), site.bulk_equivalent))
                grain = Grain(
                    self.oriented_unit_cell.copy(),
                    self.miller_index.copy(),
                    slab,
                    self.shift.copy(),
                    self.mirror_x,
                    self.mirror_y,
                    self.mirror_z,
                    self.hkl_spacing,
                )
                break
        self.bulk_repeats = br
        return grain

    def make_supercell(self, scaling_matrix: ArrayLike):
        sm = np.array(scaling_matrix)
        self.ab_scale = sm[:2]
        if len(sm) == 3:
            self.bulk_repeats = sm[2]

    @property
    def translation_vec(self) -> np.ndarray:
        return self._translation_vec

    @translation_vec.setter
    def translation_vec(self, vector: ArrayLike):
        vector = np.array(vector)
        if len(vector) == 2:
            vector = np.append(vector, 0)
        self._translation_vec[:2] = vector[:2]

    @translation_vec.setter
    def ab_translation_vec(self, vector: ArrayLike):
        vector = np.array(vector)
        if len(vector) == 2:
            vector = np.append(vector, 0)
        vector = self.base.lattice.get_cartesian_coords(vector)
        self._translation_vec[:2] = vector[:2]

    def get_structure(
        self,
        reconstruction: Optional[Callable[["Grain", Site], bool]] = None,
        sort_key: Optional[Callable[[Site], Any]] = None,
    ) -> Structure:
        np.testing.assert_allclose(
            self.oriented_unit_cell.lattice.matrix[:2],
            self.base.lattice.matrix[:2],
            err_msg="Oriented unit cell and base have different a and b vectors",
        )
        struct = Structure.from_sites(self.sites, self.charge)
        if sort_key is not None:
            struct.sort(key=sort_key)
        if reconstruction is not None:
            del_i = []
            for i, site in enumerate(struct.sites):
                if not reconstruction(self, site):
                    del_i.append(i)
            struct.remove_sites(del_i)
        if self.bonds is not None:
            lattice = self.lattice.matrix.copy()
            lattice[2] *= (lattice[2, 2] + 5.0) / lattice[2, 2]
            slab = Slab(
                Lattice(lattice),
                struct.species_and_occu,
                struct.cart_coords,
                self.miller_index.copy(),
                self.oriented_unit_cell.copy(),
                self.shift[2],
                [1, 1, 1],
                False,
                coords_are_cartesian=True,
                site_properties=struct.site_properties,
            )
            gg = GrainGenerator(
                self.oriented_unit_cell.copy(), self.miller_index.copy()
            )
            slab = gg.repair_broken_bonds(slab, self.bonds)
            struct = Structure(
                struct.lattice,
                slab.species_and_occu,
                slab.cart_coords,
                slab.charge,
                to_unit_cell=False,
                coords_are_cartesian=True,
                site_properties=slab.site_properties,
            )
            struct.translate_sites(
                range(len(struct)),
                [0, 0, -1 * struct.frac_coords.min(axis=0)[2]],
                to_unit_cell=False,
            )
        return struct

    def get_sorted_structure(
        self,
        reconstruction: Optional[Callable[["Grain", Site], bool]] = None,
    ) -> Structure:
        sorting_key = lambda site: (site.species, site.z, site.x, site.y)
        return self.get_structure(
            reconstruction=reconstruction,
            sort_key=sorting_key,
        )

    def get_slab(
        self,
        bulk_repeats: Optional[int] = None,
        reconstruction: Optional[Callable[["Grain", Site], bool]] = None,
        sort_key: Optional[Callable[[Site], Any]] = None,
    ) -> Slab:
        lattice = self.lattice.matrix.copy()
        lattice[2] *= (lattice[2, 2] + 5.0) / lattice[2, 2]
        br = self.bulk_repeats
        if bulk_repeats is not None:
            self.bulk_repeats = bulk_repeats
        slab = self.get_structure(reconstruction, sort_key)
        self.bulk_repeats = br
        return Slab(
            Lattice(lattice),
            slab.species_and_occu,
            slab.cart_coords,
            self.miller_index.copy(),
            self.oriented_unit_cell.copy(),
            self.shift[2],
            [1, 1, 1],
            False,
            coords_are_cartesian=True,
            site_properties=slab.site_properties,
        )

    @classmethod
    def from_oriented_unit_cell(
        cls,
        oriented_unit_cell: Structure,
        miller_index: ArrayLike,
        shift: float,
        mirror_x: bool = False,
        mirror_y: bool = False,
        mirror_z: bool = False,
        hkl_spacing: Optional[float] = None,
    ) -> "Grain":
        base, origin_shift = cls.base_from_oriented_unit_cell(oriented_unit_cell, shift)
        grain = cls(
            oriented_unit_cell,
            miller_index,
            base,
            origin_shift,
            mirror_x,
            mirror_y,
            mirror_z,
            hkl_spacing,
        )
        return grain

    @staticmethod
    def base_from_oriented_unit_cell(
        oriented_unit_cell: Structure, shift: float = 0.0
    ) -> Tuple[Structure, np.ndarray]:
        ouc = oriented_unit_cell.copy()
        shift = ouc.lattice.get_cartesian_coords([0, 0, shift])
        lattice = ouc.lattice.matrix.copy()
        lattice[2] /= lattice[2, 2]
        lattice[2] *= ouc.lattice.matrix[2, 2] + 5
        lattice = Lattice(lattice)
        ouc.translate_sites(
            indices=range(len(oriented_unit_cell)),
            vector=-shift,
            to_unit_cell=True,
            frac_coords=False,
        )
        try:
            ouc._sites[0].bulk_equivalent
        except AttributeError:
            sg = SpacegroupAnalyzer(ouc)
            ouc.add_site_property(
                "bulk_equivalent",
                sg.get_symmetry_dataset()["equivalent_atoms"].tolist(),
            )
        ouc.sort(key=lambda site: (round(site.z, 8), site.bulk_equivalent))
        origin_shift = ouc.cart_coords[0]
        coords = []
        for site in ouc:
            c = (
                np.round(lattice.get_fractional_coords(site.coords - origin_shift), 8)
                % 1
            )
            coords.append(c)
        origin_shift += shift
        return (
            Structure(
                lattice,
                ouc.species_and_occu,
                coords,
                ouc.charge,
                site_properties=ouc.site_properties,
            ),
            origin_shift,
        )


class GrainBoundary:
    """A grain boundary for storing grains.

    TODO:
      * Docstrings.
      * Lattice matching -> find Moire lattice.
    """

    def __init__(
        self,
        grain_0: Grain,
        grain_1: Optional[Grain] = None,
        mirror_x: Optional[bool] = None,
        mirror_y: Optional[bool] = None,
        mirror_z: Optional[bool] = None,
        translation_vec: ArrayLike = [0.0, 0.0, 0.0],
        vacuum: Optional[float] = None,
        merge_tol: Optional[float] = None,
        reconstruction: Optional[Callable[["Grain", Site], bool]] = None,
    ):
        self.grain_0 = grain_0
        self.grain_1 = grain_0.copy() if grain_1 is None else grain_1
        self.grain_1.mirror_x = (
            mirror_x if mirror_x is not None else self.grain_1.mirror_x
        )
        self.grain_1.mirror_y = (
            mirror_y if mirror_y is not None else self.grain_1.mirror_y
        )
        self.grain_1.mirror_z = (
            mirror_z if mirror_z is not None else self.grain_1.mirror_z
        )
        self.translation_vec = translation_vec
        self.vacuum = vacuum
        self.reconstruction = reconstruction
        self.merge_tol = merge_tol

    @property
    def translation_vec(self) -> np.ndarray:
        return self._translation_vec

    @translation_vec.setter
    def translation_vec(self, v: ArrayLike):
        v = np.array(v)
        if len(v) == 2:
            v = np.append(v, self.translation_vec[2])
        self.grain_1.translation_vec = v
        self._translation_vec = v

    @translation_vec.setter
    def ab_translation_vec(self, v: ArrayLike):
        v = np.array(v)
        if len(v) == 2:
            v = np.append(v, self.translation_vec[2])
        self.grain_1.ab_translation_vec = v
        self._translation_vec = np.array([*self.grain_1.translation_vec[:2], v[2]])

    def orthogonalise_c(self):
        self.grain_0.orthogonalise_c()
        self.grain_1.orthogonalise_c()

    def scan(
        self,
        na: int,
        nb: int,
        nc: int = 1,
        da: Optional[float] = None,
        db: Optional[float] = None,
        dc: Optional[float] = None,
        z_on_both_interfaces: bool = True,
    ):
        """An iterator that yeilds grainboundary Structures for scanning over the grains.

        hello.
        """
        da = 1 / na if da is None else da
        db = 1 / nb if db is None else db
        dc = 1 / nc if dc is None else dc
        a = self.grain_0.base.lattice.matrix[0] * da
        b = self.grain_0.base.lattice.matrix[1] * db
        c = self.grain_0.base.lattice.matrix[2] * dc
        tv = self.translation_vec
        vacuum = self.vacuum
        for ia, ib, ic in product(range(na), range(nb), range(nc)):
            translation_vec = tv + ia * a
            translation_vec += ib * b
            translation_vec += ic * c
            self.translation_vec = translation_vec
            if z_on_both_interfaces and vacuum is not None:
                self.vacuum = vacuum + ic * dc
            yield self.get_structure()

    def convergence(
        self,
        bulk_repeats_0: ArrayLike,
        bulk_repeats_1: Optional[ArrayLike] = None,
    ):
        bulk_repeats_0 = np.array(bulk_repeats_0)
        bulk_repeats_1 = (
            bulk_repeats_0 if bulk_repeats_1 is None else np.array(bulk_repeats_1)
        )
        for n_0, n_1 in zip(bulk_repeats_0, bulk_repeats_1):
            if n_0 == 0 or n_1 == 0:
                continue
            self.grain_0.bulk_repeats = n_0
            self.grain_1.bulk_repeats = n_1
            yield self.copy()

    def ab_supercell(self, supercell_dimensions: ArrayLike):
        self.grain_0.make_supercell(supercell_dimensions)
        self.grain_1.make_supercell(supercell_dimensions)

    def get_structure(
        self,
        sort_key: Optional[Callable[[Site], Any]] = None,
    ) -> Structure:
        vacuum = self.vacuum if self.vacuum is not None else self.translation_vec[2]
        height = (
            self.grain_0.thickness
            + self.grain_1.thickness
            + self.translation_vec[2]
            + vacuum
        )
        lattice = self.grain_0.lattice.matrix.copy()
        lattice[2] /= self.grain_0.lattice.c
        lattice[2] *= height / lattice[2, 2]
        translation_vec = self.grain_0.lattice.matrix[2].copy()
        translation_vec[2] += self.translation_vec[2]
        grain_0 = self.grain_0.get_structure(reconstruction=self.reconstruction)
        grain_1 = self.grain_1.get_structure(reconstruction=self.reconstruction)
        # THIS NEEDS LOOKING AT, WHERE IS THE PRECISION LOST?
        coords = grain_0.lattice.get_cartesian_coords(grain_1.frac_coords)
        coords[:, 2] = grain_1.cart_coords[:, 2]
        coords = np.add(coords, translation_vec)
        site_properties = {
            k: np.concatenate([v, grain_1.site_properties[k]])
            for k, v in grain_0.site_properties.items()
            if k in grain_1.site_properties
        }
        site_properties["grain"] = np.concatenate(
            [np.repeat(0, len(grain_0)), np.repeat(1, len(grain_1))]
        )
        grain_boundary = Structure(
            lattice,
            np.concatenate([grain_0.species, grain_1.species]),
            np.concatenate([grain_0.cart_coords, coords]),
            to_unit_cell=True,
            coords_are_cartesian=True,
            site_properties=site_properties,
        )
        if self.merge_tol is not None:
            grain_boundary.merge_sites(tol=self.merge_tol, mode="delete")
        if sort_key is not None:
            grain_boundary.sort(key=sort_key)
        return grain_boundary

    def get_sorted_structure(
        self,
        key: Callable[[Site], Any] = lambda s: (
            s.species.average_electroneg,
            s.species_string,
            s.z,
            s.x,
            s.y,
        ),
    ) -> Structure:
        gb = self.get_structure()
        gb.sort(key=key)
        return gb

    def copy(self):
        return GrainBoundary(
            self.grain_0.copy(),
            self.grain_1.copy(),
            None,
            None,
            None,
            self.translation_vec,
            self.vacuum,
            self.merge_tol,
            self.reconstruction,
        )


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
