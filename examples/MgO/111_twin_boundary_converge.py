from gbmaker import GrainBoundaryGenerator
from pymatgen.core import Structure

bulk = Structure.from_file(filename="./POSCAR-bulk")
gg = GrainBoundaryGenerator(bulk, [1, 1, 1])

# return an iterator for the grains
grain_boundaries = gg.get_grain_boundaries(
    mirror_z=True, symmetrize=True, merge_tol=0.1
)


# iterate through the generated grain boundaries
for gb in grain_boundaries:
    # this ensures symmetry between both boundaries for mirrored grains
    gb.orthogonalise_c()
    # Converge slab thickness
    for grain_boundary in gb.convergence(range(6)):
        grain_boundary.get_sorted_structure().to("poscar", "POSCAR")
        # run VASP
