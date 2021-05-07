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

grain.orthogonalise_c()  # this ensures symmetry between both boundaries for mirrored grains

# Generate the grain boundary with the above settings and mirror the second
# grain in z and merging atoms within 0.2 Angstrom of each other.
gb = GrainBoundary(
    grain_1=grain,
    mirror_z=True,
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
