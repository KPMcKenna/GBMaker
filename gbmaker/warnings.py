import warnings
from pymatgen.core import Structure


def formatwarning(
    message,
    catagory,
    *_,
):
    return f"{catagory.__name__}\n{message}"


warnings.formatwarning = formatwarning


class Warnings:
    @classmethod
    def UnitCell(cls, unit_cell: Structure):
        warnings.warn(f"Non-conventional unit cell supplied, using:\n{unit_cell}")
        return
