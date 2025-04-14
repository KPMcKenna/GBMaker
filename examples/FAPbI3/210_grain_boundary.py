from pymatgen.core import Structure
from gbmaker import GrainGenerator, GrainBoundary

bondlengths ={("Pb","I"): 3.2,("N","H"): 1.05,("C","H"): 1.10, ("N","C"): 1.35}

bulk = Structure.from_file(filename="./POSCAR-bulk")
gg = GrainGenerator(bulk, [2,1,0], symmetrize=False,repair=True,bonds=bondlengths)
grains = gg.get_grains()

#visualise to determine appropriate termination
for i, g in enumerate(grains):
    g.bulk_repeats=1
    g.get_sorted_structure().to(fmt="poscar", filename=f"{i}-210_grain.vasp")



