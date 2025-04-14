from gbmaker import GrainGenerator, GrainBoundary
from pymatgen.core import Structure

bulk = Structure.from_file(filename="./POSCAR-bulk")
gg = GrainGenerator(bulk, [3, 1, 0])
grain = next(gg)

# Set grain thickness to 5 * the d spacing of the [3, 1, 0] plane, a
# fractional 0.5 shift in a - b and a symmtric 2.0 spacing between grains.
grain.hkl_thickness = 5
translation_vec = [0, 0, 2]
vacuum = 2.0  # this is unnecassary as by default the z-shift is applied to both grains
# this ensures symmetry between both boundaries for mirrored grains
grain.orthogonal_c = True

# Generate the grain boundary with the above settings and mirror the second
# grain in z.
gb = GrainBoundary(
    grain_0=grain,
    mirror_z=True,
    translation_vec=translation_vec,
    vacuum=vacuum,
)
# gb.ab_translation_vec = [0.5, 0.5]


# Output the grain boundary
gb.get_sorted_structure().to(fmt="poscar", filename="310_grain_boundary.vasp")
