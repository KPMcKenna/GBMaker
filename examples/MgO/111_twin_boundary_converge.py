from gbmaker import GrainBoundaryGenerator

gbg = GrainBoundaryGenerator.from_file(
    "./POSCAR-bulk",
    [1, 1, 1],
    mirror_z=True,
    symmetrize=True,
    merge_tol=0.1,
    orthogonal_c=True,
)

# iterate through the generated grain boundaries
for gb in gbg:
    # Converge slab thickness
    for grain_boundary in gb.convergence(range(6)):
        grain_boundary.get_sorted_structure().to("poscar", "POSCAR")
        # run VASP
