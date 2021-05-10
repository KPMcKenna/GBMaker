from gbmaker import GrainGenerator, GrainBoundary
from pymatgen.core import Structure
from pymatgen.vis.structure_vtk import MultiStructuresVis

# load the bulk cell
bulk = Structure.from_file("./POSCAR-bulk")

# generate the grains
gg = GrainGenerator(bulk, [0, 1, 1])
grains = gg.get_grains()

# view the grains
view = MultiStructuresVis()
view.set_structures(grains)
view.show()

# orthogonalise the c-vector so that each boundary is symmetric
grains[0].orthogonalise_c()

# seperate the grains by 2 Angstrom
translation_vec = [0.0, 0.0, 2.0]

# create a domain wall
gb = GrainBoundary(grain_1=grains[0], mirror_y=True, translation_vec=translation_vec)
gb.grain_1.hkl_thickness = 3.5
gb.grain_2.hkl_thickness = 3.5
gb.as_structure().get_sorted_structure().to("poscar", "011_domain_wall.vasp")
gb.grain_1.to("poscar", "./grain_0.vasp")
gb.grain_2.to("poscar", "./grain_1.vasp")
