from gbmaker import GrainBoundaryGenerator


bondlengths = {("Sb", "Se"): 3.1}
gbg = GrainBoundaryGenerator.from_file(
    "./POSCAR-bulk",
    [0, 1, 1],
    mirror_z=True,
    translation_vec=[0, 0, 1.3],
    repair=True,
    bonds=bondlengths,
    symmetrize=True,
    orthogonal_c=True,
    bulk_repeats=3,
)

for i, gb in enumerate(gbg):
    gb.get_sorted_structure().to(fmt="poscar", filename=f"{i}-011_grain_boundary.vasp")
