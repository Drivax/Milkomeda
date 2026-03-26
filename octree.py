"""
octree.py -- Barnes-Hut octree for O(N log N) force calculations.

Builds a 3D octree over the particle set, then computes gravitational
accelerations with a multipole-acceptance criterion (MAC):

	s / d < theta  -> treat cell as a single point mass at its CoM.

Also provides compute_accelerations_direct: a vectorized O(N^2) numpy solver
that is much faster than the tree for small N (< ~3000).

Units: kpc, M_sun, yr  (G = 4.498e-24 kpc^3 M_sun^-1 yr^-2)
"""

from __future__ import annotations
import numpy as np

G_CONST = 4.498e-24  # kpc^3 / (M_sun * yr^2)


class OctreeNode:
	"""A single node in the Barnes-Hut octree."""

	__slots__ = (
		"center", "size",
		"mass", "com",
		"particle_idx",
		"children",
	)

	def __init__(self, center: np.ndarray, size: float):
		self.center = center
		self.size = size
		self.mass = 0.0
		self.com = np.zeros(3)
		self.particle_idx = -1
		self.children: list[OctreeNode | None] = [None] * 8

	@property
	def is_leaf(self) -> bool:
		return all(c is None for c in self.children)

	@property
	def is_empty(self) -> bool:
		return self.mass == 0.0


def _octant(pos: np.ndarray, center: np.ndarray) -> int:
	ix = int(pos[0] >= center[0])
	iy = int(pos[1] >= center[1])
	iz = int(pos[2] >= center[2])
	return ix + 2 * iy + 4 * iz


def _child_center(parent_center: np.ndarray, octant: int, half: float) -> np.ndarray:
	dx = half if (octant & 1) else -half
	dy = half if (octant & 2) else -half
	dz = half if (octant & 4) else -half
	return parent_center + np.array([dx, dy, dz])


def _insert(node: OctreeNode, idx: int, pos: np.ndarray, mass: np.ndarray,
			depth: int = 0) -> None:
	if depth > 50:
		node.com = (node.com * node.mass + pos[idx] * mass[idx]) / (node.mass + mass[idx] + 1e-300)
		node.mass += mass[idx]
		return

	if node.is_empty and node.is_leaf:
		node.particle_idx = idx
		node.mass = mass[idx]
		node.com = pos[idx].copy()
		return

	if node.is_leaf and node.particle_idx != -1:
		resident = node.particle_idx
		node.particle_idx = -1
		half = node.size / 4.0
		oct_r = _octant(pos[resident], node.center)
		if node.children[oct_r] is None:
			node.children[oct_r] = OctreeNode(
				_child_center(node.center, oct_r, half), node.size / 2.0)
		_insert(node.children[oct_r], resident, pos, mass, depth + 1)

	half = node.size / 4.0
	oct_n = _octant(pos[idx], node.center)
	if node.children[oct_n] is None:
		node.children[oct_n] = OctreeNode(
			_child_center(node.center, oct_n, half), node.size / 2.0)
	_insert(node.children[oct_n], idx, pos, mass, depth + 1)

	total = node.mass + mass[idx]
	node.com = (node.com * node.mass + pos[idx] * mass[idx]) / (total + 1e-300)
	node.mass = total


def build_tree(pos: np.ndarray, mass: np.ndarray) -> OctreeNode:
	lo = pos.min(axis=0)
	hi = pos.max(axis=0)
	center = (lo + hi) / 2.0
	size = (hi - lo).max() * 1.01
	root = OctreeNode(center, size)
	N = pos.shape[0]
	for i in range(N):
		_insert(root, i, pos, mass)
	return root


def _acc_particle(node: OctreeNode, p_pos: np.ndarray,
				  softening: float, theta: float) -> np.ndarray:
	if node.is_empty:
		return np.zeros(3)
	dr = node.com - p_pos
	dist2 = np.dot(dr, dr) + softening ** 2
	dist = np.sqrt(dist2)
	if node.is_leaf or node.size / dist < theta:
		return G_CONST * node.mass * dr / (dist2 * dist)
	else:
		acc = np.zeros(3)
		for child in node.children:
			if child is not None:
				acc += _acc_particle(child, p_pos, softening, theta)
		return acc


def compute_accelerations(pos: np.ndarray, mass: np.ndarray,
						  softening: float = 0.1, theta: float = 0.5) -> np.ndarray:
	"""Compute gravitational accelerations using Barnes-Hut O(N log N).

	Parameters
	----------
	pos       : (N, 3) positions in kpc
	mass      : (N,)   masses in M_sun
	softening : softening length in kpc (default 0.1)
	theta     : opening angle parameter (default 0.5)

	Returns
	-------
	acc : (N, 3) accelerations in kpc / yr^2
	"""
	N = pos.shape[0]
	root = build_tree(pos, mass)
	acc = np.zeros((N, 3))
	for i in range(N):
		acc[i] = _acc_particle(root, pos[i], softening, theta)
	return acc


def compute_accelerations_direct(pos: np.ndarray, mass: np.ndarray,
								  softening: float = 0.1) -> np.ndarray:
	"""Compute gravitational accelerations via direct O(N^2) numpy broadcasting.

	Much faster than the pure-Python tree for N < ~3000 because all inner
	loops execute inside numpy C code.

	Parameters
	----------
	pos       : (N, 3) positions in kpc
	mass      : (N,)   masses in M_sun
	softening : softening length in kpc (default 0.1)

	Returns
	-------
	acc : (N, 3) accelerations in kpc / yr^2
	"""
	# dr[i, j] = pos[j] - pos[i]  shape (N, N, 3)
	dr = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
	dist2 = (dr ** 2).sum(axis=2) + softening ** 2   # (N, N)
	dist3 = dist2 ** 1.5
	np.fill_diagonal(dist3, 1.0)  # diagonal of dr is zero, so self-term = 0
	# acc[i] = G * sum_j  m_j * dr[i,j] / dist3[i,j]
	acc = G_CONST * (mass[np.newaxis, :, np.newaxis] * dr / dist3[:, :, np.newaxis]).sum(axis=1)
	return acc
