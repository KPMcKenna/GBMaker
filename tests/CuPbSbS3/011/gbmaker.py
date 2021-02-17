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


def gengb(slaboriginal_grain1,slaboriginal_grain2,GB_x_shift=0,GB_y_shift=0,GB_z_shift1=0,GB_z_shift2=0,MergeOn=False,merge_tol=1.0,MirrorX=False,MirrorY=False,MirrorZ=True):
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
    if MirrorX:
      newcoord[0]=newcoord[0]*-1
    if MirrorY:
      newcoord[1]=newcoord[1]*-1
    if MirrorZ:
      newcoord[2]=(newcoord[2]-mingrain2+GB_z_shift1/2.0)*-1
    else:
      newcoord[2]=(newcoord[2]-mingrain2+(maxgrain1-mingrain1)+GB_z_shift2+GB_z_shift1/2.0)
    all_coords.append(newcoord)

  GBSlab=Structure(newlatticematrix,all_species,all_coords,coords_are_cartesian=True)

  if MergeOn:
    GBSlab.merge_sites(merge_tol,mode="delete")
  return GBSlab
  
  
