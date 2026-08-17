"""Confinement Hamiltonians as Kronecker sums over directions, plus optional potentials.
Qubit 0 is the LSB of the grid index; in 2-D, x is the low qubits and y the high ones."""
import numpy as np


def box_kinetic(N):
    """1-D hard-wall kinetic operator: 2I minus the nearest-neighbour hops, no wrap."""
    M = np.zeros((N, N))
    for i in range(N):
        M[i, i] = 2.0
        if i + 1 < N:
            M[i, i + 1] = -1.0
            M[i + 1, i] = -1.0
    return M


def potential_profile(N, u0, meff=0.19, a0=2e-10):
    """Sine-bump potential u_i = (u0/t0) sin(pi i/(N+1)), normalised by the TB parameter."""
    hbar, m0, q0 = 1.055e-34, 9.11e-31, 1.6e-19
    t0 = hbar**2 / (2 * m0 * meff * a0**2) / q0
    return (u0 / t0) * np.sin(np.pi * np.arange(1, N + 1) / (N + 1))


def _embed(A, start, m, n):
    """Embed an operator on qubits [start, start+m) into the n-qubit space."""
    return np.kron(np.eye(2**(n - start - m)), np.kron(A, np.eye(2**start)))


class Problem:
    """H = sum_d t_d M_box(d) + diag(ux), with the exact spectrum from a dense eigh.

    dims: (start, m, t) per direction, a contiguous qubit range and its TB weight."""

    def __init__(self, dims, ux=None, label=""):
        self.dims = list(dims)
        self.n = sum(m for _, m, _ in self.dims)
        self.label = label
        N = 2**self.n
        M = np.zeros((N, N))
        for start, m, t in self.dims:
            M += t * _embed(box_kinetic(2**m), start, m, self.n)
        self.ux = None if ux is None else np.asarray(ux, dtype=float)
        if self.ux is not None:
            M += np.diag(self.ux)
        self.M = M
        w, v = np.linalg.eigh(M)
        self.lam_min = float(w[0])
        g = v[:, 0]
        self.ground = g / np.linalg.norm(g)
        # exact excited-state references (for VQD metrics); columns already sorted by eigh
        self.eigvals = w
        self.eigvecs = v
        self.excited1 = v[:, 1] / np.linalg.norm(v[:, 1])
        self.gap01 = float(w[1] - w[0])


def problem_1d(n, u0=0.0):
    """1-D hard-wall box on n qubits, optional sine-bump potential of strength u0 (eV)."""
    ux = potential_profile(2**n, u0) if u0 != 0.0 else None
    return Problem([(0, n, 1.0)], ux=ux, label=f"1D box N={2**n}, u0={u0} eV")


def problem_2d(nx, ny, t1=1 / 0.19, t2=1 / 0.98):
    """Anisotropic 2-D hard-wall box (Ning Ham2D): default Si masses t1=1/0.19 (x), t2=1/0.98 (y).
       x = low qubits [0,nx), y = high qubits [nx,nx+ny)."""
    return Problem([(0, nx, t1), (nx, ny, t2)],
                   label=f"2D box {2**nx}x{2**ny}, t1={t1:.3f}, t2={t2:.3f}")


def problem_3d(nx, ny, nz, t1=1.0, t2=1.0, t3=1.0):
    """3-D hard-wall box, isotropic by default.
       x = low qubits [0,nx), y = [nx,nx+ny), z = high qubits [nx+ny,nx+ny+nz)."""
    return Problem([(0, nx, t1), (nx, ny, t2), (nx + ny, nz, t3)],
                   label=f"3D box {2**nx}x{2**ny}x{2**nz}, t=({t1:.3g},{t2:.3g},{t3:.3g})")
