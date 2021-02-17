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
gb_plane=[3,1,0]  #GB plane
slab_min_t=3.5     #slab thickness in hkl units

Symmetric=False  #whether slabs should be symmetric
Grain1_slabid=0  #grain1 id
Grain2_slabid=0  #grain2 id

#Grain2 transformation (e.g. for sym tilt GB MirrorX=False, MirrorY=False, MirrorZ=True)
MirrorX=False
MirrorY=False
MirrorZ=True

MergeOn=False    #whether or not to merge atoms if closer than merge_tol
merge_tol=1.5    #distance for which atoms are merged (if using)

vacuum=5.0       #vacuum gap to use for output of slab structures

#Parameters for gamma surface scan
Na=4
Nb=8
Nz=6
Delz_min=1.0
Delz_max=2.0
MinBond=1.5


#######################################
# MAIN CODE                           #
#######################################

file=open("GBScan.dat","w",buffering=1)

#Read and analyse bulk structure provided
prim = pymatgen.Structure.from_str(open("POSCAR-bulk").read(), fmt="poscar")
sga = SpacegroupAnalyzer(prim)
bulk=sga.get_conventional_standard_structure() #conventional needed for surface cells

#Generate surf slabs and output for visualisation
SlabGen=SlabGenerator(bulk,gb_plane,slab_min_t,slab_min_t,lll_reduce=False, center_slab=False, in_unit_planes=True, primitive=True, max_normal_search=None,reorient_lattice=True)
slabs=SlabGen.get_slabs(symmetrize=Symmetric)
for i in range(0,len(slabs)):
  slab=genslab(slabs[i],vacuum=vacuum)
  slab.get_sorted_structure().to(filename='POSCAR-slab-{}'.format(i), fmt="poscar")

#Loop over translations in a,b and z  
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
      GBSlab=gengb(slabs[Grain1_slabid],slabs[Grain2_slabid],GB_x_shift,GB_y_shift,GB_z_shift1,GB_z_shift2,MergeOn,merge_tol,MirrorX,MirrorY,MirrorZ)
      GBSlab.get_sorted_structure().to(filename='POSCAR', fmt="poscar")  
      if GBSlab.is_valid(tol=MinBond):
        #Vasp run (need INCAR, POTCAR and KPOINTS in dir)
        subprocess.check_output(["mpirun","-np","240","vasp_gam"])
        line=subprocess.check_output(["grep","rgy  w","OUTCAR"]).decode('ASCII').split()
        E=line[6]
        file.write("{:.4f} {:.4f} {:.4f} {:.6f}\n".format(float(i/Na),float((j/Nb)),float(GB_z_shift),float(E)))

file.close()
