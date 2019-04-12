#works with pymatgen Version: 2019.1.24
import numpy
import pymatgen
from pymatgen.core.surface import Structure, Slab, SlabGenerator
from pymatgen.transformations.standard_transformations import RotationTransformation

def genslab(slaboriginal,vacuum=0):
  #rotate cell so that surface normal along z
  axis_rot=numpy.cross(slaboriginal.normal,[0,0,1])
  angle=numpy.arccos(numpy.dot(slaboriginal.normal,[0,0,1]))
  SlabRot=RotationTransformation(axis_rot,angle,angle_in_radians=True)
  slab2=SlabRot.apply_transformation(slaboriginal)
  slaboriginal=slab2
  #extract coords
  sites=list(slaboriginal.sites)
  sortedzsites = sorted(sites, key=lambda site: site.coords[2])
  mingrain=sortedzsites[0].coords[2]
  maxgrain=sortedzsites[-1].coords[2]
  
  lattice=slaboriginal.lattice
  newlatticematrix=lattice.matrix.copy()
  newlatticematrix[2][2]=(maxgrain-mingrain)+vacuum
  newlatticematrix[2][1]=0
  
  all_species =[]
  all_species.extend(site.specie for site in sortedzsites)

  all_coords =[]
  for i in range(0,len(sortedzsites)):
    newcoord=sortedzsites[i].coords
    newcoord[0]=newcoord[0]
    newcoord[1]=newcoord[1]
    newcoord[2]=newcoord[2]-mingrain
    all_coords.append(newcoord)

  slab=Structure(newlatticematrix,all_species,all_coords,coords_are_cartesian=True)

  return slab


def gengb(slaboriginal_grain1,slaboriginal_grain2,GB_x_shift=0,GB_y_shift=0,GB_z_shift1=0,GB_z_shift2=0,MergeOn=False,merge_tol=1.0):
  #rotate cell so that surface normal along z
  axis_rot=numpy.cross(slaboriginal_grain1.normal,[0,0,1])
  angle=numpy.arccos(numpy.dot(slaboriginal_grain1.normal,[0,0,1]))
  SlabRot=RotationTransformation(axis_rot,angle,angle_in_radians=True)
  slab2=SlabRot.apply_transformation(slaboriginal_grain1)
  slaboriginal_grain1=slab2
  slab2=SlabRot.apply_transformation(slaboriginal_grain2)
  slaboriginal_grain2=slab2
  #extract coords
  sites=list(slaboriginal_grain1.sites)
  sortedzsitesg1 = sorted(sites, key=lambda site: site.coords[2])
  mingrain1=sortedzsitesg1[0].coords[2]
  maxgrain1=sortedzsitesg1[-1].coords[2]
  sites=list(slaboriginal_grain2.sites)
  sortedzsitesg2 = sorted(sites, key=lambda site: site.coords[2])
  mingrain2=sortedzsitesg2[0].coords[2]
  maxgrain2=sortedzsitesg2[-1].coords[2]
  
  lattice=slaboriginal_grain1.lattice
  newlatticematrix=lattice.matrix.copy()
  newlatticematrix[2][2]=(maxgrain1-mingrain1)+(maxgrain2-mingrain2)+GB_z_shift1+GB_z_shift2
  newlatticematrix[2][1]=0
  #print(newlatticematrix)
  
  all_species =[]
  all_species.extend(site.specie for site in sortedzsitesg1)
  all_species.extend(site.specie for site in sortedzsitesg2)

  all_coords =[]
  for i in range(0,len(sortedzsitesg1)):
    newcoord=sortedzsitesg1[i].coords
    newcoord[0]=newcoord[0]+GB_x_shift
    newcoord[1]=newcoord[1]+GB_y_shift
    newcoord[2]=newcoord[2]-mingrain1+(GB_z_shift1/2.0) 
    all_coords.append(newcoord)
  for i in range(0,len(sortedzsitesg2)):
    newcoord=sortedzsitesg2[i].coords
    newcoord[2]=(newcoord[2]-mingrain2+GB_z_shift1/2.0)*-1
    all_coords.append(newcoord)

  GBSlab=Structure(newlatticematrix,all_species,all_coords,coords_are_cartesian=True)

  if MergeOn:
    GBSlab.merge_sites(merge_tol,mode="delete")
  return GBSlab
  
  
#Subroutine for analysis that return some cost function
def cost(GBSlab,BulkBonds,GBRegion,Rsearch,BondTol,printanalysis=False):
#Test bonds near GB region to count number of reasonable bonds and repulsive bonds
  GBatoms=0
  c=0
  for j in range(0,len(BulkBonds)):
    BulkBonds[j][5]=0
    BulkBonds[j][6]=0
    BulkBonds[j][7]=0
  for site in GBSlab:
    #find atoms near GB at z=0
    if abs(site.coords[2])<GBRegion:
      GBatoms=GBatoms+1
      #identify NN sites and bonds
      neigb=GBSlab.get_neighbors(site, Rsearch, include_index=False, include_image=False)    
      Sp1=site.specie.name #need .name as specie is enum type
      for j in range(0,len(BulkBonds)): #count number of atoms of each type
        if ((Sp1==BulkBonds[j][1]) or (Sp1==BulkBonds[j][2])):
          BulkBonds[j][7]=BulkBonds[j][7]+1 
      #for each neighbour
      for i in range(0,len(neigb)):
        Sp2=neigb[i][0].specie.name
        #for each of the bulk bond types
        for j in range(0,len(BulkBonds)):
          if ((Sp1==BulkBonds[j][1]) and ((Sp2==BulkBonds[j][2]))) or ((Sp1==BulkBonds[j][2]) and ((Sp2==BulkBonds[j][1]))):
            if BulkBonds[j][0]=='B':
              #If reasonable bond check in range
              if (abs(neigb[i][1]-BulkBonds[j][3])/BulkBonds[j][3]) < BondTol:
                BulkBonds[j][5]=BulkBonds[j][5]+1            
                BulkBonds[j][6]=BulkBonds[j][6]+neigb[i][1]
            if BulkBonds[j][0]=='R':
              #If unreasonable bond check in range
              if neigb[i][1] < BulkBonds[j][3]:
                BulkBonds[j][5]=BulkBonds[j][5]+1

  if(printanalysis):
    print('Atoms in GB analysis region:',GBatoms )
  for j in range(0,len(BulkBonds)):
#    if(printanalysis):
#      print("Type",BulkBonds[j][0],":",BulkBonds[j][1],"-",BulkBonds[j][2],"bonds per atom = ",BulkBonds[j][5]/BulkBonds[j][7],"(bulk:",BulkBonds[j][4],")")
    if BulkBonds[j][0]=='B':
      if(printanalysis):
        print("         average bond length = ",BulkBonds[j][6]/BulkBonds[j][5],"(bulk:",BulkBonds[j][3],")")
      c=c+abs((BulkBonds[j][5]/GBatoms)-BulkBonds[j][4])

  return c  
