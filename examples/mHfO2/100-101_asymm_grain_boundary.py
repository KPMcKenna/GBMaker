from gbmaker import GrainGenerator, GrainBoundary
from pymatgen.core import Structure
#from pymatgen.vis.structure_vtk import MultiStructuresVis
import numpy as np

#1. Generate (100) grain
bulk = Structure.from_file(filename="./POSCAR-bulk")
gg = GrainGenerator(bulk, [1, 0, 0], symmetrize=False,ftol=0.02)
grains = gg.get_grains()

#visualise to determine appropriate termination
#structures=[]
#for i, g in enumerate(grains):
#    g.bulk_repeats=3
#    g.get_sorted_structure().to("poscar", f"{i}-100_grain.vasp")
#    structures.append(g.get_structure())

#vis = MultiStructuresVis()
#vis.set_structures(structures)
#vis.show()

#set grain parameters and make supercell
grain0=grains[3]
grain0.bulk_repeats=5
grain0.orthogonal_c = True
grain0.make_supercell((1,3))
grain0.get_sorted_structure().to("poscar", "100_grain.vasp")

#2. Generate (101) grain
gg = GrainGenerator(bulk, [1, 0, 1], symmetrize=False)
grains = gg.get_grains()

#visualise to determine appropriate termination
#structures=[]
#for i, g in enumerate(grains):
#    g.bulk_repeats=3
#    g.get_sorted_structure().to("poscar", f"{i}-101_grain.vasp")
#    structures.append(g.get_structure())

#set grain parameters and make supercell
grain1=grains[1]
grain1.bulk_repeats=6
grain1.orthogonal_c = True
grain1.make_supercell((1,2))
grain1.get_sorted_structure().to("poscar", "101_grain.vasp")

#3. Generate the grain boundary with the above settings
#scale to average strain to make commensurate (not sure if this is right?)
#grain0_b=grain0.lattice.matrix[1,1]
#grain1_b=grain1.lattice.matrix[1,1]
#average_b=(grain0_b+grain1_b)/2.0
#scale=average_b/grain0_b
#grain0.ab_scale=[1.0,scale]
#scale=average_b/grain1_b
#grain1.ab_scale=[1.0,scale]

gb = GrainBoundary(
    grain_0=grain0,
    grain_1=grain1,
    translation_vec=[0,0,1.6],
    vacuum=10.0
)

# Output the grain boundary
gb.get_sorted_structure().to("poscar", "100-101-mHfO2-GB.vasp")

