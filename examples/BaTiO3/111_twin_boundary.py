import numpy as np
from gbmaker import GrainGenerator, GrainBoundary
from pymatgen.core import Structure
from pymatgen.vis.structure_vtk import MultiStructuresVis

bulk = Structure.from_file(filename="./POSCAR-bulk")
gg = GrainGenerator(bulk, [1, 1, 1])
grains = list(gg.get_grains(symmetrize=True))

structures = [g.get_structure() for g in grains]
print(structures)

vis = MultiStructuresVis()
vis.set_structures(structures)
vis.show()

grain = grains[0]

# Set grain thickness to 5 * the d spacing of the [3, 1, 0] plane, a
# fractional 0.5 shift in a - b and a symmtric 2.0 spacing between grains.
grain.hkl_thickness = 3
translation_vec = [0, 0, 0]
# vacuum = 2.0  # this is unnecassary as by default the z-shift is applied to both grains
grain.orthogonal_c = True

# Generate the grain boundary with the above settings and mirror the second
# grain in z.
gb = GrainBoundary(
    grain_0=grain,
    mirror_z=True,
)

# Output the grain boundary
gb.get_sorted_structure().to("poscar", "111_BTO-twin.vasp")
