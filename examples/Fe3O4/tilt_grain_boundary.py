from gbmaker import GrainBoundaryGenerator

gbg = GrainBoundaryGenerator.from_file(
    "./POSCAR-bulk",
    [3, 3, 2],
    mirror_z=True,
    translation_vec=[0, 0, 0.0],
    symmetrize=True,
    orthogonal_c=True,
    bulk_repeats=1,
    merge_tol=0.1
)

for i, gb in enumerate(gbg):
    gb.get_sorted_structure().to("poscar", f"{i}-_grain_boundary.vasp")
