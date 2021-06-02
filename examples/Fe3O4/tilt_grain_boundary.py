from gbmaker import GrainBoundaryGenerator

gbg = GrainBoundaryGenerator.from_file(
    "./POSCAR-bulk",
    [3, 2, -2],
    mirror_z=True,
    translation_vec=[0, 0, 0.0],
    symmetrize=True,
    orthogonal_c=True,
    bulk_repeats=1,
    merge_tol=0.1,
)

for i, gb in enumerate(gbg):
    gb.get_sorted_structure().to("poscar", f"{i}_grain_boundary.vasp")

gbg = GrainBoundaryGenerator.from_file(
    "./POSCAR-bulk",
    [0, 1, 5],
    mirror_z=True,
    translation_vec=[0, 0, 0.0],
    symmetrize=True,
    orthogonal_c=True,
    bulk_repeats=1,
    merge_tol=0.1,
    relative_to_bulk_0=True,
)

for i, gb in enumerate(gbg):
    gb.get_sorted_structure().to("poscar", f"{i}-grain_boundary.vasp")
