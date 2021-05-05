from gbmaker import GrainGenerator, GrainBoundary
from pymatgen.core import Structure
from pymatgen.vis.structure_vtk import MultiStructuresVis

bulk = Structure.from_file("./POSCAR-bulk")

gg = GrainGenerator(bulk, [0, 1, 1])

grains = gg.get_grains()
grains[0].orthogonalise_c()

view = MultiStructuresVis()
view.set_structures(grains)
view.show()

translation_vec = grains[0].scaled_c_vector / grains[0].thickness
translation_vec *= 2.0

gb = GrainBoundary(grain_1=grains[0], mirror_y=True, translation_vec=translation_vec)
gb.grain_1.hkl_thickness = 3.5
gb.grain_2.hkl_thickness = 3.5
gb.as_structure().get_sorted_structure().to("poscar", "011_grainboundary.vasp")
gb.grain_1.to("poscar", "./grain_0.vasp")
gb.grain_2.to("poscar", "./grain_1.vasp")
