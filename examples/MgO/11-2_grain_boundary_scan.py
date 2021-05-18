from gbmaker import GrainGenerator, GrainBoundary
from pymatgen.core import Structure

bulk = Structure.from_file(filename="./POSCAR-bulk")
gg = GrainGenerator(bulk, [1, 1, -2])
grain = gg.get_grains(symmetrize=False)[0]

# Set grain thickness to 5 * the d spacing of the [1, 1, -2] plane, a
# 2.0 Angstrom gap between grains.
grain.hkl_thickness = 5
translation_vec = [0.0, 0.0, 2.0]
grain.orthogonalise_c()  # this ensures symmetry between both boundaries for mirrored grains

# Generate the grain boundary with the above settings and mirror the second
# grain in z.
gb = GrainBoundary(
    grain_0=grain,
    mirror_z=True,
    translation_vec=translation_vec,
)


# Scan the grain boundary
for grain_boundary in gb.scan(na=5, nb=5):
    if grain_boundary.is_valid(0.2):
        grain_boundary.get_sorted_structure().to("poscar", "POSCAR")
        # run VASP
