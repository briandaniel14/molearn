import os
import sys

import numpy as np
import torch
from openmm import LangevinIntegrator
from openmm.app import ForceField, Modeller, Simulation
from openmm.unit import femtoseconds, kelvin, picoseconds

from molearn.data.pdb_data import PDBData

sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), "src"))

from molearn.loss_functions import openmm_energy
from molearn.trainers.openmm_physics_trainer import SOFT_NB_XML
from molearn.utils import as_numpy, random_string


def simulate_murd_stability_v2(
    coords,
    atoms,
    std=None,
    mol=None,
    xml_file=None,
    soft_nb=True,
    clamp_threshold=1e8,
    clamp=False,
    platform=None,
    **kwargs,
):
    """
    Simulate MurD protein stability using OpenMM, following the conventions of OpenMM_Physics_Trainer.
    Args:
        coords: (N_atoms, 3) torch tensor or numpy array of xyz coordinates
        atoms: list of atom names (order must match coords)
        std: standardization factor (if needed)
        mol: molecule object (if needed by openmm_energy)
        xml_file: force field XML(s) or None for default
        soft_NB: use soft nonbonded force field
        clamp_threshold: clamp forces if clamp=True
        clamp: whether to clamp forces
        platform: OpenMM platform string
        **kwargs: passed to openmm_energy
    Returns:
        energy (torch.Tensor or float)
    """
    coords = as_numpy(coords)
    if coords.shape[1] != 3:  # noqa : PLR2004
        raise ValueError(f"Expected coords shape (N, 3), got {coords.shape}")

    # XML force field logic (copied from OpenMM_Physics_Trainer)

    tmp_filename = None
    xml_files = xml_file
    if xml_files is None:
        if soft_nb:
            tmp_filename = f"soft_nonbonded_{random_string()}.xml"
            with open(tmp_filename, "w") as handle:
                handle.write(SOFT_NB_XML)
            xml_files = ["amber14-all.xml", tmp_filename]
            kwargs.setdefault("remove_NB", True)
        else:
            xml_files = ["amber14-all.xml"]

    clamp_kwargs = dict(max=clamp_threshold, min=-clamp_threshold) if clamp else None
    energy = openmm_energy(
        mol,
        std,
        clamp=clamp_kwargs,
        platform=platform or ("CUDA" if torch.cuda.is_available() else "Reference"),
        atoms=atoms,
        xml_file=xml_files,
        coords=coords,
        **kwargs,
    )

    if tmp_filename is not None and os.path.exists(tmp_filename):
        os.remove(tmp_filename)

    return energy


def simulate_murd_short_md(
    coords,
    pdb_path,
    n_steps=1000,
    temperature=300,
    platform_name="Reference",
    forcefield_files=("amber14-all.xml", "amber14/tip3p.xml"),
    minimize=True,
    **kwargs,
):
    """
    Run a short OpenMM MD simulation to check stability of a conformation.
    Args:
        coords: (N_atoms, 3) torch tensor or numpy array of xyz coordinates (in nm)
        pdb_path: Path to PDB file (for topology)
        n_steps: Number of MD steps
        temperature: Simulation temperature (K)
        platform_name: OpenMM platform string
        forcefield_files: tuple of force field XML files
        minimize: Whether to minimize energy before MD
    Returns:
        initial_energy, final_energy, stable (bool)
    """
    coords = as_numpy(coords)
    if coords.shape[1] != 3:  # noqa: PLR2004
        raise ValueError(f"Expected coords shape (N, 3), got {coords.shape}")

    pdb = PDBData()
    pdb.import_pdb(pdb_path)
    pdb.fix_terminal()
    pdb.atomselect(atoms=["N", "CA", "CB", "C", "O"])
    topology = pdb.topology
    positions = coords * 0.1  # if coords are in Angstroms

    forcefield = ForceField(*forcefield_files)
    modeller = Modeller(topology, positions)
    modeller.addHydrogens(forcefield)
    system = forcefield.createSystem(topology, nonbondedMethod=None, constraints=None)
    integrator = LangevinIntegrator(
        temperature * kelvin, 1.0 / picoseconds, 0.002 * femtoseconds
    )
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    if minimize:
        simulation.minimizeEnergy()
    state = simulation.context.getState(getEnergy=True)
    initial_energy = state.getPotentialEnergy()._value

    try:
        simulation.step(n_steps)
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        final_energy = state.getPotentialEnergy()._value
        pos = state.getPositions(asNumpy=True)
        stable = np.isfinite(final_energy) and np.all(np.isfinite(pos._value))
    except Exception as e:
        print(f"Simulation failed: {e}")
        return initial_energy, None, False

    return initial_energy, final_energy, stable
