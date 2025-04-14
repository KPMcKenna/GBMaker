from gbmaker import GrainBoundaryGenerator
from pymatgen.core import Structure

bulk = Structure.from_file(filename="./POSCAR-bulk")
gb = next(
    GrainBoundaryGenerator(
        bulk,
        [1, 1, -2],
        orthogonal_c=True,
        hkl_thickness=5,
        translation_vec=[0, 0, 2.0],
        mirror_z=True,
    )
)

# Scan the grain boundary
for grain_boundary in gb.scan(na=5, nb=5):
    if grain_boundary.is_valid(0.2):
        grain_boundary.get_sorted_structure().to(fmt="poscar", filename="POSCAR")
        # run VASP
