from gbmaker import GrainBoundaryGenerator


gbg = GrainBoundaryGenerator.from_file("./POSCAR-bulk", [0, 1, 1])
bondlengths = {("Sb", "Se"): 3.1}

for gb in gbg.get_grain_boundaries(
    mirror_z=True,
    translation_vec=[0, 0, 1.3],
    repair=True,
    bonds=bondlengths,
    symmetrize=True,
):
    gb.orthogonalise_c()
    gb.grain_0.hkl_thickness = 3
    gb.grain_1.hkl_thickness = 3
    gb.ab_translation_vec = [0, 0]
    gb.get_sorted_structure().to("poscar", f"011_grain_boundry.vasp")
