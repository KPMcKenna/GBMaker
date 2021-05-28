import numpy as np
import warnings
from itertools import product
from functools import reduce
from pymatgen.core import Structure, Lattice, Site, PeriodicSite, IStructure
from pymatgen.core.surface import SlabGenerator, Slab
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.util.typing import SpeciesLike
from typing import Dict, Sequence, Any, Callable, Optional, List, Iterator
from numpy.typing import ArrayLike
from .warnings import Warnings


class GrainGenerator(SlabGenerator):
    """Class for generating Grains.

    TODO:
      * Tidy up implementation: slab -> grain, np.ndarray -> ArrayLike, etc
    """

    def __init__(
        self,
        bulk_cell: Structure,
        miller_index: ArrayLike,
        bonds: Optional[Dict[Sequence[SpeciesLike], float]] = None,
        ftol: float = 0.1,
        tol: float = 0.1,
        max_broken_bonds: int = 0,
        symmetrize: bool = False,
        repair: bool = False,
        orthogonal_c: bool = False,
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
        if unit_cell != bulk_cell:
            Warnings.UnitCell(unit_cell)
        primitive_cell = sg.get_primitive_standard_structure()
        T = sg.get_conventional_to_primitive_transformation_matrix()
        # calculate the miller index relative to the primitive cell
        primitive_miller_index = np.dot(T, miller_index)
        primitive_miller_index /= reduce(float_gcd, primitive_miller_index)
        primitive_miller_index = np.array(
            np.round(primitive_miller_index, 1), dtype=int
        )
        super().__init__(primitive_cell, primitive_miller_index, None, None)
        # use the conventional standard structure and the supplied miller index
        self.parent = unit_cell
        self.miller_index = np.array(miller_index)
        self.orthogonal_c = orthogonal_c
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

        if symmetrize:
            grains = [g for g in grains if g.can_symmetrize_surfaces(True)]

        match = StructureMatcher(
            ltol=tol,
            stol=tol,
            primitive_cell=False,
            scale=False,
        )
        self.iter = iter([grain for grain, *_ in match.group_structures(grains)])

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
            orthogonal_c=self.orthogonal_c,
        )

    def get_grains(self) -> List["Grain"]:
        """Get the differing terminations of Grains for the oriented unit cell.

        This is a similar method to get_slabs() but yeilding Grains instead of
        returning slabs. Grains are repaired before they are symmetrized and
        there is no current way to reverse this behavour.
        """
        return list(self)

    def __iter__(self) -> Iterator["Grain"]:
        return self.iter.__iter__()

    def __next__(self) -> "Grain":
        return self.iter.__next__()


class GrainBoundaryGenerator:
    """A class for generating GrainBoundary classes."""

    def __init__(
        self,
        bulk_0: Structure,
        miller_0: ArrayLike,
        bulk_1: Optional[Structure] = None,
        miller_1: Optional[ArrayLike] = None,
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
        bulk_repeats: int = 1,
        thickness: Optional[float] = None,
        hkl_thickness: Optional[float] = None,
        orthogonal_c: bool = False,
    ):
        """Initialise the generator with either one or two bulk structures.

        If a second bulk structure or second miller index is supplied then the
        second grain will be made of this different grain.

        TODO:
          - Moire lattice matching.
          - Rotation.
        """
        self.grains_0 = GrainGenerator(
            bulk_0,
            miller_0,
            bonds,
            ftol,
            tol,
            max_broken_bonds,
            symmetrize,
            repair,
            orthogonal_c,
        ).get_grains()
        for g in self.grains_0:
            if hkl_thickness is not None:
                g.hkl_thickness = hkl_thickness
            elif thickness is not None:
                g.thickness = thickness
            else:
                g.bulk_repeats = bulk_repeats

        if bulk_1 is not None or miller_1 is not None:
            bulk_1 = bulk_1 if bulk_1 is not None else bulk_0
            miller_1 = miller_1 if miller_1 is not None else miller_0
            self.grains_1 = GrainGenerator(
                bulk_1,
                miller_1,
                bonds,
                ftol,
                tol,
                max_broken_bonds,
                symmetrize,
                repair,
                orthogonal_c,
            ).get_grains()
            for g in self.grains_1:
                if hkl_thickness is not None:
                    g.hkl_thickness = hkl_thickness
                elif thickness is not None:
                    g.thickness = thickness
                else:
                    g.bulk_repeats = bulk_repeats

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

            self.gb_map = map(map_func_gs, product(self.grains_0, self.grains_1))
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

            self.gb_map = map(map_func_g, self.grains_0)

    def __iter__(self) -> Iterator["GrainBoundary"]:
        """Generate an iterator of GrainBoundaries over possible grain terminations.

        Using the arguements available to the GrainGenerator and GrainBoundary
        classes create an iterator over the available grains.
        """
        return self.gb_map.__iter__()

    def __next__(self) -> "GrainBoundary":
        return self.gb_map.__next__()

    def get_grain_boundaries(self) -> List["GrainBoundary"]:
        return list(self)

    @classmethod
    def from_file(
        cls,
        filename_0: str,
        miller_0: ArrayLike,
        filename_1: Optional[str] = None,
        miller_1: Optional[ArrayLike] = None,
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
        bulk_repeats: int = 1,
        thickness: Optional[float] = None,
        hkl_thickness: Optional[float] = None,
        orthogonal_c: bool = False,
    ) -> "GrainBoundaryGenerator":
        """Convience method for creating GrainBoundaries from files.

        Opens any file that is parsable by pymatgen.core.Structure.from_file.
        An optional second file or second miller index can be supplied to create
        a GrainBoundary with different materials and/or different oriented grains.
        """
        bulk_0 = Structure.from_file(filename_0)
        bulk_1 = None if filename_1 is None else Structure.from_file(filename_1)
        return cls(
            bulk_0,
            miller_0,
            bulk_1,
            miller_1,
            mirror_x=mirror_x,
            mirror_y=mirror_y,
            mirror_z=mirror_z,
            vacuum=vacuum,
            translation_vec=translation_vec,
            merge_tol=merge_tol,
            reconstruction=reconstruction,
            bonds=bonds,
            ftol=ftol,
            tol=tol,
            max_broken_bonds=max_broken_bonds,
            symmetrize=symmetrize,
            repair=repair,
            bulk_repeats=bulk_repeats,
            thickness=thickness,
            hkl_thickness=hkl_thickness,
            orthogonal_c=orthogonal_c,
        )


class Grain:
    """A grain class the builds upon a pymatgen Structure.

    TODO:
      * Add missing docstrings.
    """

    def __init__(
        self,
        oriented_unit_cell: Structure,
        miller_index: ArrayLike,
        mirror_x: bool = False,
        mirror_y: bool = False,
        mirror_z: bool = False,
        hkl_spacing: Optional[float] = None,
        bonds: Optional[Dict[Sequence[SpeciesLike], float]] = None,
        orthogonal_c: bool = False,
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
        hkl_spacing: The separation of the miller index planes for calculating
                     relative thickness.
        bonds: A dictionary of pairs of atomic species and their maximum bond
               length for repairing the surfaces of grains.
        orthogonal_c: Whether to orthogonalise the c-vector.
        """
        self.miller_index = np.array(miller_index)
        self.hkl_spacing = hkl_spacing
        self.oriented_unit_cell = IStructure.from_sites(oriented_unit_cell)
        self.bulk_thickness = self.oriented_unit_cell.lattice.matrix[2, 2]
        self.bulk_repeats = 1
        self.ab_scale = [1, 1]
        self._mirror = np.zeros(3, dtype=bool)
        self.mirror_x = mirror_x
        self.mirror_y = mirror_y
        self.mirror_z = mirror_z
        self._translation_vec = np.zeros(3, dtype=float)
        self._extra_thickness = (
            self.oriented_unit_cell.cart_coords.max(axis=0)[2]
            - self.oriented_unit_cell.lattice.matrix[2, 2]
        )
        self.orthogonal_c = orthogonal_c
        self.bonds = bonds

    def __len__(self) -> int:
        """The number of sites in the structure."""
        # if the attribute '_len' exists then the grain is too be symmetrized
        # and as such does not have a bulk multiple of atoms.
        try:
            return self._len
        except AttributeError:
            ouc_len = (
                self.bulk_repeats
                * np.product(self.ab_scale)
                * len(self.oriented_unit_cell)
            )
            return ouc_len

    @property
    def symmetrize(self) -> bool:
        """Whether the Grain is to be symmetrized."""
        # if the attribute '_symmetrize' has not been set then this is false.
        try:
            return self._symmetrize
        except AttributeError:
            return False

    @symmetrize.setter
    def symmetrize(self, b: bool):
        # check the surface can be symmetrized before allowing it to be set.
        if b:
            self.can_symmetrize_surfaces(True)
            if not self.symmetrize:
                warnings.warn("Cannot symmetrize surface.")
        # if trying to set false delete the attribute '_symmetrize' if it exists.
        elif self.symmetrize:
            self.__delattr__("_symmetrize")

    @property
    def lattice(self) -> Lattice:
        """The lattice of the grain (before repair or symmetrize)."""
        h = self.thickness + 5.0
        lattice = self.oriented_unit_cell.lattice.matrix.copy()
        lattice[0] *= self.ab_scale[0]
        lattice[1] *= self.ab_scale[1]
        lattice[2] *= h / lattice[2, 2]
        if self.orthogonal_c:
            lattice[2, :2] = 0
        return Lattice(lattice)

    @property
    def charge(self) -> Optional[float]:
        """Charge of the structure."""
        # if the unit cell has charge multiply that charge up with the repeats.
        try:
            chg = self.oriented_unit_cell.charge * (self.bulk_repeats + self.symmetrize)
            chg *= np.product(self.ab_scale)
        except TypeError:
            chg = None
        return chg

    @property
    def sites(self) -> List[PeriodicSite]:
        """List of all the sites in the Grain."""
        return list(self)

    def __iter__(self) -> Iterator[PeriodicSite]:
        return self.get_structure().__iter__()

    def __getitem__(self, ind: int) -> PeriodicSite:
        """Return a specific site when the Grain is indexed."""
        return self.sites[ind]

    @property
    def bonds(self) -> Optional[Dict[Sequence[SpeciesLike], float]]:
        """Dictonary of pairs of atomic species and their maximum bond length."""
        return self._bonds

    @bonds.setter
    def bonds(self, bonds: Optional[Dict[Sequence[SpeciesLike], float]]):
        """Set the maximum bond length between pairs of atoms for repairing."""
        self._bonds = bonds
        # if the surface is to be symmetrized, check that this is still possible.
        if self.symmetrize:
            self.__delattr__("_symmetrize")
            self.can_symmetrize_surfaces(True)
            if self.symmetrize:
                return
            else:
                warnings.warn("Can no longer symmetrize surfaces.")
        # calculate the extra thicknes this repairing adds to the Grain.
        self._extra_thickness = (
            self.get_structure().cart_coords.max(axis=0)[2]
            - self.bulk_thickness * self.bulk_repeats
        )

    @property
    def extra_thickness(self) -> float:
        """Thickness added by lack of periodic boundary, repairs or symmetrizing."""
        return self._extra_thickness

    @property
    def thickness(self) -> float:
        """The thickness of the grain in Angstrom."""
        bulk = (self.bulk_repeats + int(self.symmetrize)) * self.bulk_thickness
        return self.extra_thickness + bulk

    @thickness.setter
    def thickness(self, thickness: float):
        """Sets the thickness of the grain as close as possible to the supplied value.

        This method will set the thickness of the grain (in Angstrom) to the
        value supplied, or as close as possible. It does this by calculating the
        required amount of oriented unit cells to add to the grain.  If the value
        supplied is not a multiple of the bulk thickness then the smallest amount
        of bulk repeats are added to make the thickness larger than the supplied value.
        """
        thickness -= self.thickness
        n = np.ceil(thickness / self.bulk_thickness)
        self.bulk_repeats += int(n)

    @thickness.setter
    def hkl_thickness(self, thickness: float):
        """Sets the thickness of the grain as close as possible to the supplied value.

        This method will set the thickness of the grain (in hlk units) to the
        value supplied, or as close as possible. It does this by calculating the
        required amount of oriented unit cells to add to the grain.  If the value
        supplied is not a multiple of the bulk thickness then the smallest amount
        of bulk repeats are added to make the thickness larger than the supplied value.
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
        """
        self._thickness_n = int(n)

    @property
    def mirror_x(self):
        """Whether to mirror the Grain along the x-direction."""
        return self._mirror[0]

    @mirror_x.setter
    def mirror_x(self, b: bool):
        """Sets whether to mirror the Grain along the x-direction."""
        self._mirror[0] = b

    @property
    def mirror_y(self):
        """Whether to mirror the Grain along the y-direction."""
        return self._mirror[1]

    @mirror_y.setter
    def mirror_y(self, b: bool):
        """Sets whether to mirror the Grain along the y-direction."""
        self._mirror[1] = b

    @property
    def mirror_z(self):
        """Whether to mirror the Grain along the z-direction."""
        return self._mirror[2]

    @mirror_z.setter
    def mirror_z(self, b: bool):
        """Sets whether to mirror the Grain along the z-direction."""
        self._mirror[2] = b

    def copy(self) -> "Grain":
        """Create a copy of the Grain."""
        grain = Grain(
            self.oriented_unit_cell.copy(),
            self.miller_index.copy(),
            self.mirror_x,
            self.mirror_y,
            self.mirror_z,
            self.hkl_spacing,
            self.bonds,
            self.orthogonal_c,
        )
        grain.bulk_repeats = self.bulk_repeats
        grain.symmetrize = self.symmetrize
        return grain

    @property
    def orthogonal_c(self) -> bool:
        """Whether to orthogonalise the c-vector."""
        return self._orth_c

    @orthogonal_c.setter
    def orthogonal_c(self, b: bool):
        """Sets whether to orthogonalise the c-vector."""
        self._orth_c = b

    def can_symmetrize_surfaces(self, set_symmetrize: bool = False) -> bool:
        """Checks if the surfaces of the Grain can be symmetrized.

        If the grain can be symmetrized, after any optional bond repairing, this
        method returns True. If the set_symmetrize flag is passed as True then
        the Grain.symmetrize flag will be set to return value of the method.
        """
        if self.symmetrize:
            return True
        # get two repeats of the bulk so that the slab can be reduced to symmetrize.
        slab = self.get_slab(bulk_repeats=2)
        # reset the extra thickness and '_len' (this could have been called
        # from bonds whilst previously being able to symmetrize)
        self._extra_thickness = slab.cart_coords.max(axis=0)[2] - (
            2 * self.bulk_thickness
        )
        try:
            self.__delattr__("_len")
        except AttributeError:
            pass
        if slab.is_symmetric():
            return True
        slab = self.symmetrize_surfaces(slab)
        if slab is None:
            return False
        elif set_symmetrize:
            self._symmetrize = True
            self._extra_thickness = slab.cart_coords.max(axis=0)[2] - (
                2 * self.bulk_thickness
            )
            self._len = len(slab)
            return True
        else:
            return True

    def symmetrize_surfaces(self, struct: Structure) -> Optional[Structure]:
        """Attempt to symmetrize both surfaces of the grain.

        This method is a reworking of the pymatgen method for SlabGenerator:
        nonstoichiometric_symmetrized_slab
        """
        slab = Slab(
            Lattice(struct.lattice.matrix.copy()),
            struct.species_and_occu,
            struct.cart_coords,
            self.miller_index.copy(),
            self.oriented_unit_cell.copy(),
            0,
            [1, 1, 1],
            False,
            coords_are_cartesian=True,
            site_properties=struct.site_properties,
        )
        if slab.is_symmetric():
            return struct

        # set the bulk thickness to 2 this means that
        slab.sort(key=lambda site: (round(site.z, 8), site.bulk_equivalent))
        # maybe add energy at some point
        # slab.energy = init_slab.energy
        grain = None
        sites = self.oriented_unit_cell * [*self.ab_scale, 1]

        for _ in sites:
            # Keep removing sites from the TOP one by one until both
            # surfaces are symmetric or the number of sites IS EQUAL TO THE
            # NUMBER OF ATOMS IN THE ORIENTED UNIT CELL.
            slab.remove_sites([len(slab) - 1])
            # Check if the altered surface is symmetric
            if slab.is_symmetric():
                # reset the slab thickness as we have removed atoms,
                # reducing the bulk thickness.
                grain = Structure.from_sites(slab, self.charge)
                break
        return grain

    def make_supercell(self, scaling_matrix: ArrayLike):
        """Create a super cell of the Grain.

        If an array of length 3 is supplied then the third element is used to
        set the bulk repeats.
        """
        sm = np.array(scaling_matrix)
        self.ab_scale = sm[:2]
        if len(sm) == 3:
            self.bulk_repeats = sm[2]

    @property
    def translation_vec(self) -> np.ndarray:
        """A 2D vector to translate the Grain by in cartesian coordinates.

        Whilst this is a 3D array the 3rd dimension is always zero.
        """
        return self._translation_vec

    @translation_vec.setter
    def translation_vec(self, vector: ArrayLike):
        """Set a vector to translate the Grain by in cartesian coordinates.

        If a 3D translation vector is supplied only the first two dimensions are
        taken.
        """
        vector = np.array(vector)
        self._translation_vec[:2] = vector[:2]

    @translation_vec.setter
    def fractional_translation_vec(self, vector: ArrayLike):
        """Set a vector to translate the Grain by in fractional coordinates.

        If a 3D translation vector is supplied only the first two dimensions are
        taken.
        """
        vector = np.array(vector)
        if len(vector) == 2:
            vector = np.append(vector, 0)
        else:
            vector[2] = 0
        vector = self.lattice.get_cartesian_coords(vector)
        self._translation_vec[:2] = vector[:2]

    def get_structure(
        self,
        reconstruction: Optional[Callable[["Grain", Site], bool]] = None,
    ) -> Structure:
        """Get the pymatgen.core.Structure object for the Grain.

        A reconstruction function can be supplied that takes the Grain and a site
        and returns a bool specifying whether to include the site in the final
        Structure.
        """
        # if we want to symmetrize then we need to add another bulk repeat
        # this bulk repeat will be removed during the symmetrizing and ensures
        # that the thickness is above the set amount
        cart_shifts = [
            np.dot(shift, self.oriented_unit_cell.lattice.matrix)
            for shift in np.ndindex(
                (
                    *self.ab_scale,
                    self.bulk_repeats + self.symmetrize,
                ),
            )
        ]
        sites = []
        cart_coords = self.oriented_unit_cell.cart_coords + self.translation_vec
        if self.symmetrize:
            lattice = self.lattice.matrix.copy()
            lattice[2] *= (
                1 + self.oriented_unit_cell.lattice.matrix[2, 2] / lattice[2, 2]
            )
            lattice = Lattice(lattice)
        else:
            lattice = self.lattice
        for coord, site in zip(cart_coords, self.oriented_unit_cell):
            for shift in cart_shifts:
                c = np.round(lattice.get_fractional_coords(coord + shift), 14) % 1
                sites.append(
                    PeriodicSite(
                        site.species,
                        c,
                        lattice,
                        to_unit_cell=False,
                        coords_are_cartesian=False,
                        properties=site.properties,
                    )
                )

        # the wrapping in pymatgen is succeptible to floating point errors and we
        # need to make sure that the 'bottom' layer stays on the bottom
        struct = self.repair_broken_bonds(Structure.from_sites(sites, self.charge))
        if self.symmetrize:
            struct = self.symmetrize_surfaces(struct)
            if struct is None:
                raise TypeError
        # after we have built, repaired and symmetrized the grain we can now
        # mirror in cartesian directions.
        if np.any(self._mirror):
            mirror = np.power([-1, -1, -1], self._mirror)
            height = self.thickness / self.lattice.matrix[2, 2]
            z_shift = self.lattice.matrix[2] * height
            z_shift *= -int(self.mirror_z) * mirror
            c = struct.cart_coords
            c *= mirror
            c += z_shift
            c = np.round(self.lattice.get_fractional_coords(c), 12) % 1
            struct = Structure(
                self.lattice.matrix.copy(),
                struct.species_and_occu,
                c,
                struct.charge,
                False,
                False,
                False,
                struct.site_properties,
            )
        if reconstruction is not None:
            del_i = []
            for i, site in enumerate(struct.sites):
                if not reconstruction(self, site):
                    del_i.append(i)
            struct.remove_sites(del_i)
        return struct

    def get_sorted_structure(
        self,
        reconstruction: Optional[Callable[["Grain", Site], bool]] = None,
        key: Callable[[Site], Any] = lambda s: (
            s.species.average_electroneg,
            s.species_string,
            s.z,
            s.x,
            s.y,
        ),
    ) -> Structure:
        """Get a sorted pymatgen.core.Structure object for the Grain.

        A reconstruction function can be supplied that takes the Grain and a site
        and returns a bool specifying whether to include the site in the final
        Structure.
        """
        struct = self.get_structure(reconstruction=reconstruction)
        struct.sort(key=key)
        return struct

    def get_slab(
        self,
        bulk_repeats: Optional[int] = None,
        reconstruction: Optional[Callable[["Grain", Site], bool]] = None,
    ) -> Slab:
        """Get a pymatgen.core.surface.Slab object for the Grain.

        A reconstruction function can be supplied that takes the Grain and a site
        and returns a bool specifying whether to include the site in the final
        Structure.

        It can often be useful to have a bulk repeats that is different to the
        value stored in the Grain, most commonly one extra repeat, and so a
        temporary value can be supplied.
        """
        br = self.bulk_repeats
        if bulk_repeats is not None:
            self.bulk_repeats = bulk_repeats
        slab = self.get_structure(reconstruction)
        self.bulk_repeats = br
        return Slab(
            Lattice(slab.lattice.matrix.copy()),
            slab.species_and_occu,
            slab.cart_coords,
            self.miller_index.copy(),
            self.oriented_unit_cell.copy(),
            0,
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
        shift: float = 0.0,
        mirror_x: bool = False,
        mirror_y: bool = False,
        mirror_z: bool = False,
        hkl_spacing: Optional[float] = None,
        bonds: Optional[Dict[Sequence[SpeciesLike], float]] = None,
        orthogonal_c: bool = False,
    ) -> "Grain":
        """Create a Grain from an oriented unit cell and a c-vector fractional shift.

        This method will prepare the oriented unit cell for the Grain and should
        be used to create the class, rather than using the init method. The
        oriented unit cell is decorated with 'bulk equivalent' to identify atoms,
        rotated so that the ab-plane lies in the xy-plane and that the a-vector
        runs parallel to the x-direction.
        """
        # try and orthogonalise the structure
        ouc = oriented_unit_cell.copy()
        if "bulk_equivalent" not in ouc.site_properties:
            sg = SpacegroupAnalyzer(ouc)
            ouc.add_site_property(
                "bulk_equivalent",
                sg.get_symmetry_dataset()["equivalent_atoms"].tolist(),
            )
        # first rotate the oriented unit cell so that the miller index is along z
        R = rotation(np.cross(*ouc.lattice.matrix[:2]))
        symmop = SymmOp.from_rotation_and_translation(rotation_matrix=R)
        ouc.apply_operation(symmop)
        # then rotate the oriented unit cell so that the a-vector lies along x
        theta = np.arccos(ouc.lattice.matrix[0, 0] / ouc.lattice.a)
        theta *= -1 if ouc.lattice.matrix[0, 1] > 0 else 1
        symmop = SymmOp.from_axis_angle_and_translation(
            [0, 0, 1],
            theta,
            True,
        )
        ouc.apply_operation(symmop)
        ouc = orthogonalise(ouc)
        origin_shift = ouc.lattice.get_cartesian_coords([0, 0, shift])
        ouc.translate_sites(
            indices=range(len(oriented_unit_cell)),
            vector=-origin_shift,
            to_unit_cell=True,
            frac_coords=False,
        )
        ouc.sort(key=lambda site: (round(site.z, 8), site.bulk_equivalent))
        origin_shift = ouc.frac_coords[0]
        ouc.translate_sites(
            indices=range(len(oriented_unit_cell)),
            vector=-origin_shift,
            to_unit_cell=False,
            frac_coords=True,
        )
        # lets try and remove the floating point error from the structure
        lattice = np.round(ouc.lattice.matrix, 15)
        ouc = Structure(
            lattice,
            ouc.species_and_occu,
            np.mod(np.round(ouc.frac_coords, 15), 1),
            ouc.charge,
            False,
            False,
            False,
            ouc.site_properties,
        )
        grain = cls(
            ouc,
            miller_index,
            mirror_x,
            mirror_y,
            mirror_z,
            hkl_spacing,
            bonds,
            orthogonal_c,
        )
        return grain

    def repair_broken_bonds(self, struct: Structure) -> Structure:
        """Repair bonds on the surface of the Grain.

        Very heavily taken from pymatgen.core.surface.SlabGenerator.repair_broken_bonds
        with tweaks to improve lookoup speed and ensure that the 'lowest' atom
        is located at z = 0.

        TODO:
            Use bulk equivalent property to specify the coordination of the atom
        """
        if self.bonds is None:
            return struct
        else:
            struct = Structure.from_sites(struct)
        for pair in self.bonds.keys():
            blength = self.bonds[pair]
            # First lets determine which element should be the
            # reference (center element) to determine broken bonds.
            # e.g. P for a PO4 bond. Find integer coordination
            # numbers of the pair of elements wrt to each other
            cn_dict = {}
            for i, el in enumerate(pair):
                cnlist = set()
                for site in self.oriented_unit_cell:
                    poly_coord = 0
                    if site.species_string == el:
                        for nn in self.oriented_unit_cell.get_neighbors(site, blength):
                            if nn[0].species_string == pair[i - 1]:
                                poly_coord += 1
                        cnlist.add(poly_coord)
                cn_dict[el] = cnlist
            # We make the element with the higher coordination our reference
            if max(cn_dict[pair[0]]) > max(cn_dict[pair[1]]):
                element1, element2 = pair
            else:
                element2, element1 = pair
            for i, site in enumerate(struct):
                # Determine the coordination of our reference
                if site.species_string == element1:
                    poly_coord = 0
                    for neighbor in struct.get_neighbors(site, blength):
                        poly_coord += 1 if neighbor.species_string == element2 else 0
                    # suppose we find an undercoordinated reference atom
                    if poly_coord not in cn_dict[element1]:
                        # We get the reference atom of the broken bonds
                        # (undercoordinated), move it to the other surface
                        struct = self.move_to_other_side(struct, [i])
                        # find its NNs with the corresponding
                        # species it should be coordinated with
                        neighbors = struct.get_neighbors(
                            struct[i], blength, include_index=True
                        )
                        tomove = [
                            nn[2]
                            for nn in neighbors
                            if nn[0].species_string == element2
                        ]
                        tomove.append(i)
                        # and then move those NNs along with the central
                        # atom back to the other side of the slab again
                        struct = self.move_to_other_side(struct, tomove)
        struct.translate_sites(
            range(len(struct)),
            [0, 0, -struct.frac_coords.min(axis=0)[2]],
            to_unit_cell=False,
        )
        return struct

    def move_to_other_side(
        self,
        init_struct: Structure,
        index: ArrayLike,
    ) -> Structure:
        """Moves a collection of atoms to the other side of the Structure.

        The collection of atoms are identified by their index within the Structure
        and are shifted by integer multiples of the oriented unit cell c-vector.
        As such is unsuitable for symmetrized or modified Grain surfaces.
        """
        struct = init_struct.copy()
        # in small cells the periodic repeat of a site is sometimes included
        # this needs to be removed otherwise the site is moved twice.
        index = np.unique(index)
        weights = [s.species.weight for s in struct]
        com = np.average(struct.frac_coords, weights=weights, axis=0)

        # Sort the index of sites based on which side they are on
        top_site_index = [i for i in index if struct[i].frac_coords[2] > com[2]]
        bottom_site_index = [i for i in index if struct[i].frac_coords[2] < com[2]]

        s = (
            self.oriented_unit_cell.lattice.matrix[2]
            * (self.bulk_repeats + int(self.symmetrize))
            * self.bulk_thickness
            / self.oriented_unit_cell.lattice.matrix[2, 2]
        )
        s *= np.ceil(s[2] / self.bulk_thickness) * self.bulk_thickness / s[2]

        # Translate sites to the opposite surfaces
        struct.translate_sites(
            top_site_index,
            -s,
            frac_coords=False,
            to_unit_cell=False,
        )
        struct.translate_sites(
            bottom_site_index,
            s,
            frac_coords=False,
            to_unit_cell=False,
        )
        return struct


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
        """Initialise the GrainBoundary structure."""
        self.grain_0 = grain_0
        self.grain_1 = grain_0.copy() if grain_1 is None else grain_1
        # We do not want to overwrite the mirror bools in the Grain.
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
    def lattice(self) -> Lattice:
        """The Lattice of the GrainBoundary."""
        vacuum = self.vacuum if self.vacuum is not None else self.translation_vec[2]
        height = (
            self.grain_0.thickness
            + self.grain_1.thickness
            + self.translation_vec[2]
            + vacuum
        )
        lattice = self.grain_0.lattice.matrix.copy()
        lattice[2] *= height / lattice[2, 2]
        return Lattice(lattice)

    @property
    def grain_offset(self) -> np.ndarray:
        """The offset of the second Grain due to the first Grain and z-translation."""
        return (
            (self.grain_0.thickness + self.translation_vec[2])
            / self.lattice.matrix[2, 2]
            * self.lattice.matrix[2]
        )

    @property
    def translation_vec(self) -> np.ndarray:
        """Vector to translate the 2nd Grain by."""
        return self._translation_vec

    @translation_vec.setter
    def translation_vec(self, v: ArrayLike):
        """Set a vector to translate the 2nd Grain by in cartesian coordinates."""
        v = np.array(v)
        if len(v) == 2:
            v = np.append(v, self.translation_vec[2])
        self.grain_1.translation_vec = v
        self._translation_vec = v

    @translation_vec.setter
    def fractional_translation_vec(self, v: ArrayLike):
        """Set a vector to translate the 2nd Grain by in fractional coordinates."""
        v = np.array(v)
        if len(v) == 2:
            c = self.lattice.get_fractional_coords(self.translation_vec)[2]
            v = np.append(v, 0)
        else:
            c = v[2]
            v[2] = 0
        v = self.lattice.get_cartesian_coords(v)
        self.grain_1.translation_vec = v
        self._translation_vec = [*v[:2], c]

    def orthogonalise_c(self):
        """Orthogonalise the c-vector of the Grains to make symmetric boundaries."""
        self.grain_0.orthogonal_c = True
        self.grain_1.orthogonal_c = True

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
        """An iterator that yeilds grain boundary Structures for scanning over the Grains.

        Unless supplied the step sizes (da, db, dc), in Angstrom, are calculated
        from the supplied number of steps (na, nb, nc) as equal divsions along
        respective lattice vector. By default any steps in the c-direction are
        applied to both interfaces in the GrainBoundary.
        """
        da = 1 / na if da is None else da
        db = 1 / nb if db is None else db
        dc = 1 / nc if dc is None else dc
        a = self.lattice.matrix[0] * da
        b = self.lattice.matrix[1] * db
        c = self.lattice.matrix[2] * dc
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
        bulk_repeats_0: Sequence[int],
        bulk_repeats_1: Optional[Sequence[int]] = None,
    ):
        """An iterator that yeilds GrainBoundaries with variable bulk repeats.

        Sequence of integers are used to create GrainBoundaries with varying bulk
        repeats in each Grain, useful for converging Grain thickness. By default
        the same sequence is used for both Grains in the boundary, however a
        second sequence can be supplied. Note the sequences are zipped together
        so willed be truncated if not the same length.
        """
        bulk_repeats_1 = bulk_repeats_0 if bulk_repeats_1 is None else bulk_repeats_1
        for n_0, n_1 in zip(bulk_repeats_0, bulk_repeats_1):
            if n_0 == 0 or n_1 == 0:
                continue
            self.grain_0.bulk_repeats = n_0
            self.grain_1.bulk_repeats = n_1
            yield self.copy()

    def ab_supercell(self, supercell_dimensions: ArrayLike):
        """Create a supercell from the supplied array.

        If 3 dimensions are supplied then the 3rd is set as the bulk repeats for
        both Grains.
        """
        self.grain_0.make_supercell(supercell_dimensions)
        self.grain_1.make_supercell(supercell_dimensions)

    def get_structure(
        self,
    ) -> Structure:
        """Get the pymatgen.core.Structure object for the GrainBoundary."""
        grain_0 = self.grain_0.get_structure(reconstruction=self.reconstruction)
        grain_1 = self.grain_1.get_structure(reconstruction=self.reconstruction)
        coords = grain_0.lattice.get_cartesian_coords(grain_1.frac_coords)
        coords[:, 2] = grain_1.cart_coords[:, 2]
        coords = np.add(coords, self.grain_offset)
        site_properties = {
            k: np.concatenate([v, grain_1.site_properties[k]])
            for k, v in grain_0.site_properties.items()
            if k in grain_1.site_properties
        }
        site_properties["grain"] = np.concatenate(
            [np.repeat(0, len(grain_0)), np.repeat(1, len(grain_1))]
        )
        grain_boundary = Structure(
            self.lattice,
            np.concatenate([grain_0.species, grain_1.species]),
            np.concatenate([grain_0.cart_coords, coords]),
            to_unit_cell=True,
            coords_are_cartesian=True,
            site_properties=site_properties,
        )
        if self.merge_tol is not None:
            grain_boundary.merge_sites(tol=self.merge_tol, mode="delete")
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
        """Get a sorted pymatgen.core.Structure object for the GrainBoundary."""
        gb = self.get_structure()
        gb.sort(key=key)
        return gb

    def copy(self):
        """Copy the GrainBoundary object."""
        return GrainBoundary(
            self.grain_0.copy(),
            self.grain_1.copy(),
            None,
            None,
            None,
            self.translation_vec.copy(),
            self.vacuum,
            self.merge_tol,
            self.reconstruction,
        )


def orthogonalise_c(s: Structure) -> Structure:
    """Force the orthogonalisation of the c-vector."""
    lattice = s.lattice.matrix.copy()
    lattice[2, :2] = 0
    return Structure(
        lattice,
        s.species_and_occu,
        s.cart_coords,
        s.charge,
        False,
        False,
        True,
        s.site_properties,
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
