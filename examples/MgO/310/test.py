from gbmaker import GrainGenerator, GrainBoundary
from pymatgen.core import Structure

bulk = Structure.from_file(filename="./POSCAR-bulk")
gg = GrainGenerator(bulk, [3, 1, 0])
grain = gg.get_grains(symmetrize=False)[0]

# Set grain thickness to 3.5 * the d spacing of the [3, 1, 0] plane, a
# fractional 0.5 shift in a and an asymmtric 1.9 and 2.1 spacing between grains.
grain.hkl_thickness = 3.5
translation_vec = grain.lattice.matrix[0] * 0.5
translation_vec[2] += 1.9
vacuum = 2.1

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
gb.grain_2.to("poscar", "grain_2.vasp")

# Output the grain boundary
gb.as_structure().get_sorted_structure().to("poscar", "grain_boundary.vasp")
