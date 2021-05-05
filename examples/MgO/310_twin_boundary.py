import numpy as np
from gbmaker import GrainGenerator, GrainBoundary
from pymatgen.core import Structure

bulk = Structure.from_file(filename="./POSCAR-bulk")
gg = GrainGenerator(bulk, [3, 1, 0])
grain = gg.get_grains(symmetrize=False)[0]

# Set grain thickness to 5 * the d spacing of the [3, 1, 0] plane, a
# fractional 0.5 shift in a - b and a symmtric 2.0 spacing between grains.
grain.hkl_thickness = 5
translation_vec = 0.5 * np.subtract(*grain.lattice.matrix[:2])
translation_vec[2] += 2.0
vacuum = 2.0  # this is unnecassary as by default the z-shift is applied to both grains
grain.orthogonalise_c()  # this ensures symmetry between both boundaries for mirrored grains

# Generate the grain boundary with the above settings and mirror the second
# grain in z.
gb = GrainBoundary(
    grain_1=grain,
    mirror_z=True,
    translation_vec=translation_vec,
    vacuum=vacuum,
)

# Output the two slabs
gb.grain_1.to("poscar", "grain_1.vasp")
gb.grain_1.oriented_unit_cell.to("poscar", "unit_cell.vasp")
gb.grain_2.to("poscar", "grain_2.vasp")

# Output the grain boundary
gb_struct = gb.as_structure()
gb_struct.get_sorted_structure().to("poscar", "grain_boundary.vasp")
