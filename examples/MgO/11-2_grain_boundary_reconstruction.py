import numpy as np
from gbmaker import Grain, GrainBoundary
from pymatgen.core import Structure, Site

ouc = Structure.from_file(filename="./POSCAR-112")
grain_1 = Grain.from_oriented_unit_cell(ouc, [1, 1, -2])

# Set grain thickness to 5 * the d spacing of the [1, 1, -2] plane, a
# fractional 0.5 shift in a - b and a symmtric 2.0 spacing between grains.
grain_1.orthogonalise_c()  # this ensures symmetry between both boundaries for mirrored grains
grain_2 = grain_1.copy()
grain_2.mirror_z = True
grain_2.translate_grain(-grain_2.frac_coords[-1])
Z_SHIFT = 3.0

# Define the reconstruction function
def reconstruction(gb: Grain, site: Site) -> bool:
    m = 3 ** 0.5
    x = site.z - 0.01
    y = site.x
    limit_1 = y <= m * x
    limit_2 = y >= -m * x + gb.lattice.a
    limit_3 = y >= m * x + 0.5 * gb.lattice.a - m * gb.thickness
    limit_4 = y <= -m * x + 0.5 * gb.lattice.a + m * gb.thickness
    in_grain = (limit_1 or limit_2) and (limit_3 and limit_4)
    return in_grain


# Generate the grain boundary with the above settings and mirror the second
# grain in z.
for i in range(1, 5):
    print(i)
    g1 = grain_1.copy()
    g2 = grain_2.copy()
    g1.make_supercell([i, 1, 6 + i])
    g2.make_supercell([i, 1, 6 + i])
    translation_vec = np.zeros(3)
    translation_vec[2] += Z_SHIFT - (3 ** 0.5 * g1.lattice.a / 6)
    gb = GrainBoundary(
        grain_1=g1.reconstruct(reconstruction),
        grain_2=g2.reconstruct(reconstruction),
        translation_vec=translation_vec,
        mirror_z=True,
    )
    gb.as_structure().get_sorted_structure().to("poscar", f"{i}_11-2_gb.vasp")
