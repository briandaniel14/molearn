import os
import sys
import numpy as np
import torch
import random
import string
import MDAnalysis as mda

def random_string(length=32):
    '''
    generate a random string of arbitrary characters. Useful to generate temporary file names.

    :param length: length of random string
    '''
    return ''.join(random.choice(string.ascii_letters)
                    for n in range(length))


def as_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.data.cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        return np.array(tensor)
    
def convert_dcd_to_pdb(pdb_file, dcd_file, output_file):
    u = mda.Universe(pdb_file, dcd_file)
    u.atoms.write(output_file, frames='all')
    print(f"Wrote {len(u.trajectory)} frames to {output_file}")

class ShutUp:
    
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self._stdout
        