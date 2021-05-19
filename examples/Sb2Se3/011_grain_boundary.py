import numpy as np
from gbmaker import GrainGenerator, GrainBoundary, GrainBoundaryGenerator
from pymatgen.core import Structure

bondlengths = {("Sb", "Se"): 3.1}

bulk = Structure.from_file(filename="./POSCAR-bulk")
gg = GrainGenerator(bulk, [0, 1, 1])
grain = list(gg.get_grains(bonds=bondlengths, symmetrize=False, repair=True))[3]

# Set grain thickness to 3 * the d spacing of the [0, 1, 1] plane, a
# fractional 0.5 shift in a - b and a symmtric 2.0 spacing between grains.
grain.hkl_thickness = 2
translation_vec = [0, 0, 5.0]
vacuum = 5.0  # this is unnecassary as by default the z-shift is applied to both grains
grain.orthogonalise_c()  # this ensures symmetry between both boundaries for mirrored grains

# Generate the grain boundary with the above settings and mirror the second
# grain in z.
gb = GrainBoundary(
    grain_0=grain,
    mirror_z=True,
    translation_vec=translation_vec,
    vacuum=vacuum,
)
gb.ab_translation_vec = [0.0, 0.5152]

# Output the grain boundary
gb.get_sorted_structure().to("poscar", "011_grain_boundary.vasp")

graincomp = list(
    GrainBoundaryGenerator(bulk, [0, 1, 1]).get_grain_boundaries(
        mirror_z=True, translation_vec=[0, 0, 5]
    )
)[3]
graincomp.ab_translation_vec = [0.0, 0.5152]
graincomp.get_sorted_structure().to("poscar", "011_grain_boundary_comp.vasp")
