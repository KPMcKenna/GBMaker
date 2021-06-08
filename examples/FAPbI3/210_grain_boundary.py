from pymatgen.core import Structure
from gbmaker import GrainGenerator, GrainBoundary

bondlengths = {("N", "H"): 1.1, ("C", "H"): 1.15, ("N", "C"): 1.35}
bonds = {
    "N": {"H": 1.1, "C": 1.35},
    "H": {"C": 1.15},
}

bulk = Structure.from_file(filename="./POSCAR-bulk")
gg = GrainGenerator(bulk, [2, 1, 0], symmetrize=False, repair=True, bonds=bondlengths)

# visualise to determine appropriate termination
for i, g in enumerate(gg):
    g.get_sorted_structure().to("poscar", f"{i}_210_grain.vasp")
