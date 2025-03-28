from gbmaker import GrainGenerator, GrainBoundary
from pymatgen.core import Structure

bulk = Structure.from_file(filename="./POSCAR-bulk")
gg = GrainGenerator(bulk, [1, 0, 0])
grains = gg.get_grains()

for i, g in enumerate(grains):
    g.get_sorted_structure().to(fmt="poscar", filename=f"{i}-POSCAR.vasp")

grain = grains[0]

# Set grain thickness to 5 * the d spacing of the [1, 1, 1] plane, a
# fractional 0.25 shift in the a-vector and a symmtric 2.0 spacing between grains.
grain.hkl_thickness = 3

# Generate the grain boundary with the above settings
gb = GrainBoundary(grain)
# shift the second grain relative to the lattice basis
gb.fractional_translation_vec = [0.0, 0.5, 0]
gb.translation_vec += (
    2.0 / gb.grain_0.lattice.matrix[2, 2]
) * gb.grain_0.lattice.matrix[2]

# Output the grain boundary
gb.get_sorted_structure().to(fmt="poscar", filename="100_stacking_fault.vasp")
