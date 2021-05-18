from gbmaker import GrainGenerator, GrainBoundary
from pymatgen.core import Structure

bulk = Structure.from_file(filename="./POSCAR-bulk")
gg = GrainGenerator(bulk, [1, 1, 1])
grains = gg.get_grains()

grain = grains[0]

# Set grain thickness to 5 * the d spacing of the [1, 1, 1] plane, a
# fractional 0.25 shift in the a-vector and a symmtric 2.0 spacing between grains.
grain.hkl_thickness = 5

# Generate the grain boundary with the above settings
gb = GrainBoundary(grain)
# shift the second grain relative to the lattice basis
gb.ab_translation_vec = [0.25, 0, 0]
gb.translation_vec += (
    2.0 / gb.grain_0.lattice.matrix[2, 2]
) * gb.grain_0.lattice.matrix[2]

# Output the grain boundary
gb.get_sorted_structure().to("poscar", "111_stacking_fault.vasp")
