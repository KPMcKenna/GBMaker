from gbmaker import GrainGenerator, GrainBoundary
from pymatgen.core import Structure
from pymatgen.vis.structure_vtk import MultiStructuresVis

# load the bulk cell
bulk = Structure.from_file("./POSCAR-bulk")

# generate the grains
gg = GrainGenerator(bulk, [0, 1, 1], orthogonal_c=True)
grains = gg.get_grains()

# view the grains
view = MultiStructuresVis()
view.set_structures(grains)
view.show()

# orthogonalise the c-vector so that each boundary is symmetric
grain = grains[0]
grain.hkl_thickness = 3.5

# seperate the grains by 2 Angstrom
translation_vec = [0.0, 0.0, 2.0]

# create a domain wall
gb = GrainBoundary(grain_0=grain, mirror_y=True, translation_vec=translation_vec)
gb.get_sorted_structure().to("poscar", "011_domain_wall.vasp")
gb.grain_0.get_sorted_structure().to("poscar", "./grain_0.vasp")
gb.grain_1.get_sorted_structure().to("poscar", "./grain_1.vasp")
