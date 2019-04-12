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

Symmetric=False  #whether slabs should be symmetric
Grain1_slabid=1  #grain1 id
Grain2_slabid=5  #grain2 id

MergeOn=False    #whether or not to merge atoms if closer than merge_tol
merge_tol=1.5    #distance for which atoms are merged (if using)

vacuum=5.0       #vacuum gap to use for output of slab structures

#Parameters for gamma surface scan (no opt)
Na=2
Nb=2
Nz=2
Delz_min=1.0
Delz_max=1.1
MinBond=1.5


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

#Loop over translations in a,b and z  
print('GRAIN BOUNDARY GAMMA SURFACE SCAN:')
print('GB plane: ',gb_plane)
print('Grain 1 id: ',Grain1_slabid)
print('Grain 2 id: ',Grain2_slabid)
print('Del_a  Del_b  Del_z  E')
latticematrix=slab.lattice.matrix  
for i in range(0, Na):
  for j in range(0,Nb):
    for k in range(0,Nz):
      GB_x_shift=(i/Na)*latticematrix[0][0]+(j/Nb)*latticematrix[1][0]
      GB_y_shift=(i/Na)*latticematrix[0][1]+(j/Nb)*latticematrix[1][1]
      GB_z_shift=(((Delz_max-Delz_min)/(Nz-1))*k)+Delz_min
      GB_z_shift1=GB_z_shift
      GB_z_shift2=GB_z_shift
      
      #Construct and output POSCAR for GB
      GBSlab=gengb(slabs[Grain1_slabid],slabs[Grain2_slabid],GB_x_shift,GB_y_shift,GB_z_shift1,GB_z_shift2,MergeOn,merge_tol)
      GBSlab.get_sorted_structure().to(filename='POSCAR-GB', fmt="poscar")  
      if GBSlab.is_valid(tol=MinBond):
        #Vasp run (need INCAR, POTCAR and KPOINTS in dir)
        subprocess.run(["cp","POSCAR-GB","POSCAR"])
        subprocess.check_output(["mpirun","-np","12","vasp_gam"])
        line=subprocess.check_output(["grep","rgy  w","OUTCAR"]).decode('ASCII').split()
        E=line[6]
        print("{:.4f} {:.4f} {:.4f} {:.6f}".format(float(i/Na),float((j/Nb)),float(GB_z_shift),float(E)))
