import os
import sys

sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), "src"))
from molearn.data import PDBData

data = PDBData()
data.import_pdb(["../data/cleaned_aligned_structure.pdb"])
data.atomselect(atoms=["N", "CA", "CB", "C", "O"])
data.write_pdb("cleaned_aligned_structure_selected.pdb")
