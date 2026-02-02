import sys, os

sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), "src"))

import MDAnalysis as mda

def convert_dcd_to_pdb(pdb_file, dcd_file, output_file):
    u = mda.Universe(pdb_file, dcd_file)
    u.atoms.write(output_file, frames='all')
    print(f"Wrote {len(u.trajectory)} frames to {output_file}")

# Convert both


if __name__ == "__main__":
    # print('ran 24')
    # main2()
    # print('ran 34')
    print('65')
    convert_dcd_to_pdb(
    "../data/cleaned_aligned_structure.pdb",
    "../data/aligned_murd_closed_npt_prod_downsampled.dcd",
    "../data/full_murd_closed.pdb"
    )
    print('71')
    convert_dcd_to_pdb(
    "../data/cleaned_aligned_structure.pdb",
    "../data/aligned_murd_open_npt_prod_downsampled.dcd",
    "../data/full_murd_open.pdb"
    )
    print('77')