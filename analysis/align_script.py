"""
Align crystal structures to a common reference frame using MDAnalysis.

Removes rigid-body translations and rotations so the CNN autoencoder
learns conformational differences rather than spatial orientation.

Usage:
    python align_structures.py --input_dir /path/to/pdbs --ref reference.pdb --output_dir /path/to/aligned
"""

from __future__ import annotations

import argparse
from pathlib import Path

import MDAnalysis as MDA  # noqa : N817
from MDAnalysis.analysis import align


def align_structures(
    input_dir: Path,
    ref_path: Path,
    output_dir: Path,
    select: str = "name CA",
) -> None:
    """
    Align all PDB files in input_dir to a reference structure.

    The rotation matrix is computed by minimising the RMSD over the
    atom selection (default: Cα backbone atoms), then applied to all
    atoms in each structure.

    Parameters
    ----------
    input_dir : Path
        Directory containing the PDB files to align.
    ref_path : Path
        Path to the reference PDB structure.
    output_dir : Path
        Directory to write the aligned PDB files.
    select : str
        MDAnalysis selection string for the atoms used to compute the
        optimal superposition. Default is "name CA" (Cα atoms).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ref = MDA.Universe(str(ref_path))
    print(f"Reference: {ref_path.name}  ({ref.atoms.n_atoms} atoms)")
    print(f"Aligning on selection: '{select}'")
    print(f"  -> {ref.select_atoms(select).n_atoms} atoms in selection\n")

    pdb_files = sorted(input_dir.glob("*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found in {input_dir}")

    for pdb in pdb_files:
        mobile = MDA.Universe(str(pdb))

        # alignto() aligns a single frame: computes the optimal rotation
        # on the 'select' atoms, then applies the transformation to all atoms
        old_rmsd, new_rmsd = align.alignto(
            mobile,
            ref,
            select=select,
            match_atoms=True,
        )

        out_path = output_dir / pdb.name
        mobile.atoms.write(str(out_path))

        print(f"  {pdb.name:>40s}  RMSD: {old_rmsd:.3f} -> {new_rmsd:.3f} Å")

    print(f"\nAligned {len(pdb_files)} structures -> {output_dir}")


def align_trajectory(
    topology: Path,
    trajectory: Path,
    ref_path: Path,
    output_path: Path,
    select: str = "name CA",
) -> None:
    """
    Align an entire trajectory to a reference structure using AlignTraj.

    Use this if your conformations are stored as frames in a single
    trajectory file (e.g. .dcd, .xtc, .trr) rather than separate PDBs.

    Parameters
    ----------
    topology : Path
        Topology file (e.g. .pdb, .psf, .prmtop).
    trajectory : Path
        Trajectory file (e.g. .dcd, .xtc).
    ref_path : Path
        Path to the reference structure.
    output_path : Path
        Path to write the aligned trajectory.
    select : str
        MDAnalysis selection string for superposition atoms.
    """
    ref = MDA.Universe(str(ref_path))
    mobile = MDA.Universe(str(topology), str(trajectory))

    print(f"Reference: {ref_path.name}  ({ref.atoms.n_atoms} atoms)")
    print(f"Trajectory: {trajectory.name}  ({len(mobile.trajectory)} frames)")
    print(f"Aligning on selection: '{select}'")
    print(f"  -> {ref.select_atoms(select).n_atoms} atoms in selection\n")

    alignment = align.AlignTraj(
        mobile,
        ref,
        select=select,
        filename=str(output_path),
        match_atoms=True,
    )
    alignment.run()

    # Per-frame RMSD values stored in alignment.results.rmsd
    rmsds = alignment.results.rmsd
    print(f"  Mean RMSD after alignment: {rmsds.mean():.3f} Å")
    print(f"  Max  RMSD after alignment: {rmsds.max():.3f} Å")
    print(f"\nAligned trajectory -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align protein structures to a reference frame."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- Mode 1: align individual PDB files ---
    pdb_parser = subparsers.add_parser("pdbs", help="Align a directory of PDB files.")
    pdb_parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing PDB files to align.",
    )
    pdb_parser.add_argument(
        "--ref",
        type=Path,
        required=True,
        help="Reference PDB structure to align against.",
    )
    pdb_parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write aligned PDB files.",
    )
    pdb_parser.add_argument(
        "--select",
        type=str,
        default="name CA",
        help="Atom selection for superposition (default: 'name CA').",
    )

    # --- Mode 2: align a trajectory file ---
    traj_parser = subparsers.add_parser("trajectory", help="Align a trajectory file.")
    traj_parser.add_argument(
        "--topology",
        type=Path,
        required=True,
        help="Topology file (.pdb, .psf, .prmtop).",
    )
    traj_parser.add_argument(
        "--trajectory",
        type=Path,
        required=True,
        help="Trajectory file (.dcd, .xtc, .trr).",
    )
    traj_parser.add_argument(
        "--ref",
        type=Path,
        required=True,
        help="Reference structure to align against.",
    )
    traj_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the aligned trajectory.",
    )
    traj_parser.add_argument(
        "--select",
        type=str,
        default="name CA",
        help="Atom selection for superposition (default: 'name CA').",
    )

    args = parser.parse_args()

    if args.mode == "pdbs":
        align_structures(args.input_dir, args.ref, args.output_dir, args.select)
    elif args.mode == "trajectory":
        align_trajectory(
            args.topology,
            args.trajectory,
            args.ref,
            args.output,
            args.select,
        )
