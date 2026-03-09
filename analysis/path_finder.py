"""A* pathfinding in 2D latent spaces, minimizing DOPE scores."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
import torch

from molearn.analysis.analyser import MolearnAnalysis


@dataclass(order=True)
class Node:
    """Node for A* search."""

    f_score: float  # g_score + h_score (for priority queue ordering)
    coord_2d: tuple[float, float]  # Position in 2D visualization space
    latent_code: np.ndarray  # Full latent code
    g_score: float = 0.0  # Actual cost so far (cumulative DOPE)
    h_score: float = 0.0  # Heuristic cost (Euclidean distance to goal)
    parent: Node | None = None  # For path reconstruction


def _euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance in 2D."""
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def get_dope_score(
    latent_code: np.ndarray,
    MA: MolearnAnalysis,  # noqa: N803
    inverse_mapping: Callable[[np.ndarray], np.ndarray],
    refine: bool = True,
) -> float:
    """Compute DOPE score for a single latent code.

    Parameters
    ----------
    latent_code : np.ndarray
        Shape (latent_dim,)
    MA : MolearnAnalysis
        Analysis object with network and dataset
    inverse_mapping : callable
        Already in latent space (not needed for decode, but kept for API consistency)
    refine : bool
        Refine structure before scoring

    Returns
    -------
    float
        DOPE score for this structure
    """
    with torch.no_grad():
        latent_tensor = torch.tensor(latent_code, dtype=torch.float32).unsqueeze(0)
        decoded = MA.network.decode(latent_tensor).cpu().numpy()[0]

    decoded_scaled = decoded * MA.stdval + MA.meanval
    dope_scores = MA.get_all_dope_score(
        decoded_scaled.reshape(1, *decoded_scaled.shape), refine=refine
    )
    return float(dope_scores[0])


def _get_neighbors_in_grid(
    coord_2d: tuple[float, float],
    grid_size: float = 0.2,
    diagonal: bool = True,
) -> list[tuple[float, float]]:
    """Get neighboring 2D coordinates in a grid.

    Parameters
    ----------
    coord_2d : tuple[float, float]
        Current 2D position
    grid_size : float
        Step size in each direction
    diagonal : bool
        Include diagonal neighbors (8-connectivity) or only cardinal (4-connectivity)

    Returns
    -------
    list[tuple[float, float]]
        List of neighbor coordinates
    """
    x, y = coord_2d
    neighbors = [
        (x + grid_size, y),
        (x - grid_size, y),
        (x, y + grid_size),
        (x, y - grid_size),
    ]
    if diagonal:
        neighbors.extend(
            [
                (x + grid_size, y + grid_size),
                (x + grid_size, y - grid_size),
                (x - grid_size, y + grid_size),
                (x - grid_size, y - grid_size),
            ]
        )
    return neighbors


def find_path(
    start_encoded: np.ndarray,
    end_encoded: np.ndarray,
    MA: MolearnAnalysis,  # noqa: N803
    mapping: Callable[[np.ndarray], np.ndarray],
    inverse_mapping: Callable[[np.ndarray], np.ndarray],
    *,
    refine: bool = True,
    grid_size: float = 0.2,
    heuristic_weight: float = 1.0,
    max_iterations: int = 10000,
) -> tuple[list[np.ndarray], dict]:
    """Find a path between two encoded states using A*, minimizing DOPE scores.

    Parameters
    ----------
    start_encoded : np.ndarray
        Starting latent code, shape (latent_dim,)
    end_encoded : np.ndarray
        Ending latent code, shape (latent_dim,)
    MA : MolearnAnalysis
        Analysis object with network and datasets
    mapping : (N, latent_dim) -> (N, 2)
        Forward map to 2D visualization space (e.g. pca.transform)
    inverse_mapping : (N, 2) -> (N, latent_dim)
        Inverse map back to latent space (e.g. pca.inverse_transform)
    refine : bool
        Refine structures before DOPE scoring
    grid_size : float
        Step size for neighbors in 2D space
    heuristic_weight : float
        Weight for heuristic vs. actual cost (higher = greedier)
    max_iterations : int
        Maximum number of nodes to explore

    Returns
    -------
    path : list[np.ndarray]
        List of latent codes from start to end
    info : dict
        Metadata: iterations, nodes_expanded, path_dope_cost
    """
    # Map to 2D space
    start_2d = tuple(mapping(start_encoded.reshape(1, -1))[0])
    end_2d = tuple(mapping(end_encoded.reshape(1, -1))[0])

    # Initialize start node
    start_dope = get_dope_score(start_encoded, MA, inverse_mapping, refine=refine)
    start_heuristic = _euclidean_distance(start_2d, end_2d) * heuristic_weight

    start_node = Node(
        f_score=start_dope + start_heuristic,
        coord_2d=start_2d,
        latent_code=start_encoded,
        g_score=start_dope,
        h_score=start_heuristic,
        parent=None,
    )

    # A* search
    open_set: list[Node] = [start_node]
    closed_set = set()
    visited_coords_2d = {}  # coord_2d -> best g_score for that coord

    iterations = 0
    nodes_expanded = 0

    while open_set and iterations < max_iterations:
        iterations += 1
        current = heappop(open_set)

        # Check if we reached the goal (within a small tolerance in 2D space)
        dist_to_goal = _euclidean_distance(current.coord_2d, end_2d)
        if dist_to_goal < grid_size * 1.5:
            # Reconstruct path
            path = []
            node: Node | None = current
            while node is not None:
                path.append(node.latent_code)
                node = node.parent
            path.reverse()

            total_cost = sum(
                get_dope_score(code, MA, inverse_mapping, refine=refine)
                for code in path
            )

            return path, {
                "iterations": iterations,
                "nodes_expanded": nodes_expanded,
                "path_dope_cost": total_cost,
                "success": True,
            }

        coord_key = (round(current.coord_2d[0], 6), round(current.coord_2d[1], 6))
        if coord_key in closed_set:
            continue
        closed_set.add(coord_key)
        nodes_expanded += 1

        # Generate neighbors
        for neighbor_2d in _get_neighbors_in_grid(
            current.coord_2d, grid_size=grid_size
        ):
            neighbor_key = (round(neighbor_2d[0], 6), round(neighbor_2d[1], 6))
            if neighbor_key in closed_set:
                continue

            # Inverse map to latent space
            neighbor_latent = inverse_mapping(np.array(neighbor_2d).reshape(1, -1))[0]

            # Compute cost
            neighbor_dope = get_dope_score(
                neighbor_latent, MA, inverse_mapping, refine=refine
            )
            g_score = current.g_score + neighbor_dope

            # Only add if this is a better path or first visit
            if (
                neighbor_key not in visited_coords_2d
                or g_score < visited_coords_2d[neighbor_key]
            ):
                visited_coords_2d[neighbor_key] = g_score

                h_score = _euclidean_distance(neighbor_2d, end_2d) * heuristic_weight
                f_score = g_score + h_score

                neighbor_node = Node(
                    f_score=f_score,
                    coord_2d=neighbor_2d,
                    latent_code=neighbor_latent,
                    g_score=g_score,
                    h_score=h_score,
                    parent=current,
                )
                heappush(open_set, neighbor_node)

    # No path found
    return [], {
        "iterations": iterations,
        "nodes_expanded": nodes_expanded,
        "path_dope_cost": float("inf"),
        "success": False,
    }
