from gbmaker import GrainBoundaryGenerator

gbg = GrainBoundaryGenerator.from_file(
    "./POSCAR-bulk",
    [0, 1, 5],
    mirror_z=True,
    translation_vec=[0, 0, 0.0],
    symmetrize=True,
    orthogonal_c=True,
    bulk_repeats=1,
    merge_tol=1.2,
    relative_to_bulk_0=True,
)

for i, gb in enumerate(gbg):
    gb.get_sorted_structure().to(fmt="poscar", filename=f"{i}-grain_boundary.vasp")
