from gbmaker import GrainBoundaryGenerator
from pymatgen.core import Structure

bulk = Structure.from_file(filename="./POSCAR-bulk")
print("gbg")
gbg = GrainBoundaryGenerator(
    bulk,
    [1, 1, 1],
    mirror_z=True,
    symmetrize=True,
    merge_tol=0.1,
    orthogonal_c=True,
    translation_vec=[0, 0, 2],
)

# iterate through the generated grain boundaries
print("gb")
gbg = gbg.get_grain_boundaries()
print(gbg)
for gb in gbg:
    # Converge slab thickness
    print("grain_boundary")
    print(gb.grain_0.thickness, gb.grain_1.thickness)
    for grain_boundary in gb.convergence(range(6)):
        grain_boundary.get_sorted_structure().to("poscar", "POSCAR")
        # run VASP
