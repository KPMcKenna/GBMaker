#works with pymatgen Version: 2019.1.24
import numpy
import pymatgen
import subprocess
from pymatgen.core.surface import Structure, Slab, SlabGenerator
from pymatgen.transformations.standard_transformations import RotationTransformation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import os
import math
from gbmaker import *


#########################
# JOB SPECIFICATION     #
#########################

#Note: intial bulk structure to be provided as POSCAR-bulk in working directory

#GB definition
gb_plane=[1,1,0] #GB plane
slab_min_t=1     #slab thickness in hkl units
GB_a_shift=0.0   #relative grain shift in a parallel to GB
GB_b_shift=0.0   #relative grain shift in b parallel to GB
GB_z_shift1=1.4  #separation in z between teminating plabes for GB 1
GB_z_shift2=1.4  #separation in z between teminating plabes for GB 2

Symmetric=False  #whether slabs should be symmetric
Grain1_slabid=1  #grain1 id
Grain2_slabid=5  #grain2 id

MergeOn=False    #whether or not to merge atoms if closer than merge_tol
merge_tol=1.5    #distance for which atoms are merged (if using)

vacuum=5.0       #vacuum gap to use for output of slab structures

#Bonding analysis (optional)
GBRegion=5.0     #region +/- around GB1 at z=0.0 to analyse
Rsearch=2.7      #max radius for bond searching
BondTol=0.3      #percentage tolerance for identification of type B bonds
BulkBonds=[]
#Bond type | Species1 | Species2 | Equilibrium/minimum bond lenth | Coordination | Num bonds (for analysis only) | sum of bond lengths | atom type counter
BulkBonds.append(["B","As","S",2.291,4,0,0,0]) #B - represents reasonable bond
BulkBonds.append(["B","Cu","S",2.309,4,0,0,0])  


#######################################
# MAIN CODE                           #
#######################################

#Read and analyse bulk structure provided
prim = pymatgen.Structure.from_str(open("POSCAR-bulk").read(), fmt="poscar")
sga = SpacegroupAnalyzer(prim)
bulk=sga.get_conventional_standard_structure() #conventional needed for surface cells
print('STRUCTURE PROVIDED:')
print(bulk)
print()

#Generate surf slabs and output for visualisation
SlabGen=SlabGenerator(bulk,gb_plane,slab_min_t,slab_min_t,lll_reduce=False, center_slab=False, in_unit_planes=True, primitive=True, max_normal_search=None,reorient_lattice=True)
slabs=SlabGen.get_slabs(symmetrize=Symmetric)
print('SURFACE SLABS:')
print('Num slabs: ',len(slabs))
for i in range(0,len(slabs)):
  print('Slab ', i, ', symmetric: ',slabs[0].is_symmetric())
  slab=genslab(slabs[i],vacuum=vacuum)
  slab.get_sorted_structure().to(filename='POSCAR-slab-{}'.format(i), fmt="poscar")
print()


#Construct and output POSCAR for GB
print('GRAIN BOUNDARY CONSTRUCTION:')
latticematrix=slab.lattice.matrix  
GB_x_shift=GB_a_shift*latticematrix[0][0]+GB_b_shift*latticematrix[1][0]
GB_y_shift=GB_a_shift*latticematrix[0][1]+GB_b_shift*latticematrix[1][1]
GBSlab=gengb(slabs[Grain1_slabid],slabs[Grain2_slabid],GB_x_shift,GB_y_shift,GB_z_shift1,GB_z_shift2,MergeOn,merge_tol)
print('Composition: ', GBSlab.composition.formula)
print('Reduced composition: ', GBSlab.composition.reduced_formula)
c=cost(GBSlab,BulkBonds,GBRegion,Rsearch,BondTol,printanalysis=True)
GBSlab.get_sorted_structure().to(filename='POSCAR-GB', fmt="poscar")  
