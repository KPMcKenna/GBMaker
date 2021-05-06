from gbmaker import GrainGenerator, GrainBoundary
from pymatgen.core import Structure

bulk = Structure.from_file(filename="./POSCAR-bulk")
gg = GrainGenerator(bulk, [1, 1, 1])
grains = gg.get_grains()

grain = grains[0]

# Set grain thickness to 3 * the d spacing of the [1, 1, 1] plane, a
# fractional 0.5 shift in a - b and a symmtric 2.0 spacing between grains.
grain.hkl_thickness = 5
translation_vec = grain.lattice.matrix[0].copy() * 0.25
translation_vec[2] += 2.0

# Generate the grain boundary with the above settings and mirror the second
# grain in z.
gb = GrainBoundary(
    grain_1=grain,
    translation_vec=translation_vec,
)

# Output the two slabs
gb.grain_1.to("poscar", "grain_1.vasp")
gb.grain_1.oriented_unit_cell.to("poscar", "unit_cell.vasp")
gb.grain_2.to("poscar", "grain_2.vasp")

# Output the grain boundary
gb_struct = gb.as_structure()
gb_struct.get_sorted_structure().to("poscar", "111_stacking_fault.vasp")
