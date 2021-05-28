from gbmaker import GrainBoundaryGenerator

# Generate the grain boundary with the above settings
gb = GrainBoundaryGenerator.from_file(
    "./POSCAR-bulk",
    [1, 1, 1],
    hkl_thickness=5,
).get_grain_boundaries()[0]
# shift the second grain relative to the lattice basis
gb.fractional_translation_vec = [0.25, 0, 0]
gb.translation_vec += (
    2.0 / gb.grain_0.lattice.matrix[2, 2]
) * gb.grain_0.lattice.matrix[2]

# Output the grain boundary
gb.get_sorted_structure().to("poscar", "111_stacking_fault.vasp")
