from gbmaker import Grain, GrainGenerator, GrainBoundary
from pymatgen.core import Structure, Site

# Read grain from an oriented unit cell
bulk = Structure.from_file(filename="./POSCAR-bulk")
grain = GrainGenerator(bulk, [1, 1, -2]).get_grains()[0]

# orthogonalise the grain and make a mirrored copy, translated so that there is
# an atom at (0, 0, 0)
grain.orthogonalise_c()  # this ensures symmetry between both boundaries for mirrored grains

# Define the reconstruction function
def reconstruction(gb: Grain, site: Site) -> bool:
    # {110} planes run at 60 deg to the c-axis (11-2) and 30 deg to the a-axis (110)
    m = 3 ** 0.5
    # as viewed down the (111) the c-axis(11-2) is our x-direction and the a-axis (110)
    # is our y-direction
    x0 = site.z + 0.2
    x1 = site.z - 0.2
    y = site.x
    # define limits that will create {110}-facetted (11-2) grains
    limit_1 = y <= m * x0
    limit_2 = y >= -m * x0 + gb.lattice.a
    limit_3 = y >= m * x1 + 0.5 * gb.lattice.a - m * gb.thickness
    limit_4 = y <= -m * x1 + 0.5 * gb.lattice.a + m * gb.thickness
    # is our atom within the limits of the grain
    in_grain = (limit_1 or limit_2) and (limit_3 and limit_4)
    return in_grain


gb = GrainBoundary(grain_0=grain, mirror_z=True, reconstruction=reconstruction)

# shift the second lattice to make the grains symmetric
a_shift = -gb.grain_0.lattice.a / 2

# reconstructing the grains is a nonreversible step (for now) so we must copy the
# grains and perform all the adjustments before the reconstruction.
for i in range(1, 5):
    gb.ab_supercell([i, 1, i + 6])
    translation_vec = [a_shift, 0, 0]
    translation_vec[2] += 2.5 - ((3 ** 0.5) * gb.grain_0.lattice.a / 6)
    gb.translation_vec = translation_vec
    gb.get_structure().to("poscar", f"{i}_11-2_gb.vasp")
