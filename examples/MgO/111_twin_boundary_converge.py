from gbmaker import GrainGenerator, GrainBoundary
from pymatgen.core import Structure
from pymatgen.vis.structure_vtk import MultiStructuresVis

bulk = Structure.from_file(filename="./POSCAR-bulk")
gg = GrainGenerator(bulk, [1, 1, 1])
grains = gg.get_grains(symmetrize=True)

view = MultiStructuresVis()
view.set_structures(grains)
view.show()

grain = grains[0]

# Set grain thickness to 3 * the d spacing of the [1, 1, 1] plane, a
# fractional 0.5 shift in a - b and a symmtric 2.0 spacing between grains.
grain.hkl_thickness = 3
translation_vec = [0.0, 0.0, 0.0]
grain.orthogonalise_c()  # this ensures symmetry between both boundaries for mirrored grains

# Generate the grain boundary with the above settings and mirror the second
# grain in z.
gb = GrainBoundary(
    grain_1=grain,
    mirror_z=True,
    translation_vec=translation_vec,
    merge_tol=0.2,
)

# Output the two slabs
gb.grain_1.to("poscar", "grain_1.vasp")
gb.grain_1.oriented_unit_cell.to("poscar", "unit_cell.vasp")
gb.grain_2.to("poscar", "grain_2.vasp")

# Output the grain boundary
gb_struct = gb.as_structure()
gb_struct.get_sorted_structure().to("poscar", "111_twin_boundary.vasp")

# Converge slab thickness
for grain_boundary in gb.convergence(range(6)):
    grain_boundary.get_sorted_structure().to("poscar", "POSCAR")
    # run VASP
