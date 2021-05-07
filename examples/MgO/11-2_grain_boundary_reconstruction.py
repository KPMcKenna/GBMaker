import numpy as np
from gbmaker import Grain, GrainBoundary
from pymatgen.core import Structure, Site

# Read grain from an oriented unit cell
ouc = Structure.from_file(filename="./POSCAR-112")
grain_1 = Grain.from_oriented_unit_cell(ouc, [1, 1, -2])

# orthogonalise the grain and make a mirrored copy, translated so that there is
# an atom at (0, 0, 0)
grain_1.orthogonalise_c()  # this ensures symmetry between both boundaries for mirrored grains
grain_2 = grain_1.copy()
grain_2.mirror_z = True
grain_2.translate_grain(-grain_2.frac_coords[-1])

# set the seperation of the grains
Z_SHIFT = 3.0

# Define the reconstruction function
def reconstruction(gb: Grain, site: Site) -> bool:
    # {110} planes run at 60 deg to the c-axis (11-2) and 30 deg to the a-axis (110)
    m = 3 ** 0.5
    # as viewed down the (111) the c-axis(11-2) is our x-direction and the a-axis (110)
    # is our y-direction
    x = site.z - 0.01
    y = site.x
    # define limits that will create {110}-facetted (11-2) grains
    limit_1 = y <= m * x
    limit_2 = y >= -m * x + gb.lattice.a
    limit_3 = y >= m * x + 0.5 * gb.lattice.a - m * gb.thickness
    limit_4 = y <= -m * x + 0.5 * gb.lattice.a + m * gb.thickness
    # is our atom within the limits of the grain
    in_grain = (limit_1 or limit_2) and (limit_3 and limit_4)
    return in_grain


# reconstructing the grains is a nonreversible step (for now) so we must copy the
# grains and perform all the adjustments before the reconstruction.
for i in range(1, 5):
    print(i)
    g1 = grain_1.copy()
    g2 = grain_2.copy()
    g1.make_supercell([i, 1, 6 + i])
    g2.make_supercell([i, 1, 6 + i])
    translation_vec = np.zeros(3)
    translation_vec[2] += Z_SHIFT - (3 ** 0.5 * g1.lattice.a / 6)
    # mirror z must match the current mirror z of the 2nd grain else it will be
    # re-mirrored
    gb = GrainBoundary(
        grain_1=g1.reconstruct(reconstruction),
        grain_2=g2.reconstruct(reconstruction),
        translation_vec=translation_vec,
        mirror_z=True,
    )
    gb.as_structure().get_sorted_structure().to("poscar", f"{i}_11-2_gb.vasp")
