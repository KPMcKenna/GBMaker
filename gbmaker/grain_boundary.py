import numpy as np
from itertools import product
from pymatgen.core import Structure, Lattice, Site
from typing import Dict, Sequence, Any, Callable, Optional, List, Iterator
from numpy.typing import ArrayLike
from pymatgen.util.typing import SpeciesLike
from .grain import Grain, GrainGenerator


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
        average_lattice: bool = False,
        vacuum: Optional[float] = None,
        translation_vec: ArrayLike = [0.0, 0.0, 0.0],
        merge_tol: Optional[float] = None,
        reconstruction: Optional[Callable[[Grain, Site], bool]] = None,
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
        relative_to_bulk_0: bool = False,
        relative_to_bulk_1: bool = False,
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
            relative_to_bulk_0,
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
                relative_to_bulk_1,
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
                    average_lattice=average_lattice,
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
                    average_lattice=average_lattice,
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
        """Return a list containing all the generated GrainBoundaries."""
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
        average_lattice: bool = False,
        vacuum: Optional[float] = None,
        translation_vec: ArrayLike = [0.0, 0.0, 0.0],
        merge_tol: Optional[float] = None,
        reconstruction: Optional[Callable[[Grain, Site], bool]] = None,
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
        relative_to_bulk_0: bool = False,
        relative_to_bulk_1: bool = False,
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
            average_lattice=average_lattice,
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
            relative_to_bulk_0=relative_to_bulk_0,
            relative_to_bulk_1=relative_to_bulk_1,
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
        average_lattice: bool = False,
        translation_vec: ArrayLike = [0.0, 0.0, 0.0],
        vacuum: Optional[float] = None,
        merge_tol: Optional[float] = None,
        reconstruction: Optional[Callable[[Grain, Site], bool]] = None,
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
        # If either grain is orthogonal both must be orthogonal_c
        if self.grain_0.orthogonal_c or self.grain_1.orthogonal_c:
            self.orthogonalise_c()
        self.translation_vec = translation_vec
        self.vacuum = vacuum
        self.reconstruction = reconstruction
        self.merge_tol = merge_tol
        self.average_lattice = average_lattice

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
        if self.average_lattice:
            lattice[0] /= self.grain_0.lattice.a
            lattice[0] *= (self.grain_0.lattice.a + self.grain_1.lattice.a) / 2
            lattice[1] /= self.grain_0.lattice.b
            lattice[1] *= (self.grain_0.lattice.b + self.grain_1.lattice.b) / 2
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
        coords_0 = self.lattice.get_cartesian_coords(grain_0.frac_coords)
        coords_0[:, 2] = grain_0.cart_coords[:, 2]
        coords_1 = self.lattice.get_cartesian_coords(grain_1.frac_coords)
        coords_1[:, 2] = grain_1.cart_coords[:, 2]
        coords_1 = np.add(coords_1, self.grain_offset)
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
            np.concatenate([coords_0, coords_1]),
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
            self.average_lattice,
            self.translation_vec.copy(),
            self.vacuum,
            self.merge_tol,
            self.reconstruction,
        )
