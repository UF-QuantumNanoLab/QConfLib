"""Qiskit key MSB..LSB = grid index
"""
import numpy as np
from qiskit import QuantumCircuit
from .ansatz import QFT_LNN, increment


# ---------------- decode helpers ----------------
def _p(counts, key_q, shots):
    """P(bitstring) in measured-clbit order c0 c1 ... (Qiskit keys are little-endian)."""
    tot = 0; L = len(key_q)
    for s, c in counts.items():
        if len(s) == L and s[::-1] == key_q:
            tot += c
    return tot / shots


def _ixZ(counts, shots):
    return sum((1 if s[-1] == '0' else -1) * c for s, c in counts.items()) / shots


def _i0xZ(counts, shots, m):
    tot = 0
    for s, c in counts.items():
        if s[:-1] == '0' * (m - 1):
            tot += (1 if s[-1] == '0' else -1) * c
    return tot / shots


def sd_dir(ansatz, n, start, m, ex, shots):
    """Sparse Decomposition (SD) (Choi et al.); O(m) circuits."""
    val = 0.0
    for j in range(m):
        mm = m - j
        qc = QuantumCircuit(n, mm)
        qc.append(ansatz.to_gate(label='U'), qc.qubits)
        for i in range(m - j - 1):
            qc.cx(start + i + 1, start + i)
        qc.h(start + m - 1 - j)
        qc.measure(range(start, start + mm), range(mm))
        counts = ex.counts(qc, shots)
        if j == m - 1:
            val += _p(counts, '0', shots) - _p(counts, '1', shots)
        else:
            tail = '0' * (mm - 2)
            val += _p(counts, tail + '10', shots) - _p(counts, tail + '11', shots)
    return 2 - val


def fd_dir(ansatz, n, start, m, ex, shots):
    """Fourier Diagonalization (FD) (Liu et al.) adapted to Dirichlet via M_box = -A_periodic + W;
       QFT Hadamard test + mcx-increment wrap correction."""
    anc = n
    qc = QuantumCircuit(n + 1, 1)
    qc.append(ansatz.to_gate(label='U'), qc.qubits[:n])
    QFT_LNN(qc, m, start); qc.h(anc)
    for idx in range(m):
        qc.cp((2 * np.pi / (2**m)) * (2**idx), anc, start + m - 1 - idx)
    qc.h(anc); qc.measure(anc, 0)
    c = ex.counts(qc, shots)
    A_per = 2 * (c.get('0', 0) - c.get('1', 0)) / shots - 2
    qc2 = QuantumCircuit(n, m)
    qc2.append(ansatz.to_gate(label='U'), qc2.qubits)
    increment(qc2, m, start); qc2.h(start)
    qc2.measure(range(start, start + m), range(m))
    W = _i0xZ(ex.counts(qc2, shots), shots, m)
    return -A_per + W


def po_dir(ansatz, n, start, m, ex, shots):
    """Permutation Operator (PO) (Sato et al.): 2 circuits (even bonds; shifted odd bonds + wrap).
    """
    c1 = QuantumCircuit(n, 1)
    c1.append(ansatz.to_gate(label='U'), c1.qubits)
    c1.h(start); c1.measure(start, 0)
    c2 = QuantumCircuit(n, m)
    c2.append(ansatz.to_gate(label='U'), c2.qubits)
    increment(c2, m, start); c2.h(start)
    c2.measure(range(start, start + m), range(m))
    cnt1 = ex.counts(c1, shots); cnt2 = ex.counts(c2, shots)
    return 2 - _ixZ(cnt1, shots) - _ixZ(cnt2, shots) + _i0xZ(cnt2, shots, m)


DIR_VALUE = {'SD': sd_dir, 'FD': fd_dir, 'PO': po_dir}
METHODS = list(DIR_VALUE)


def density_from_counts(counts, n, shots):
    p = np.zeros(2**n)
    for s, c in counts.items():
        p[int(s, 2)] += c / shots
    return p


def density(ansatz, n, ex, shots):
    qc = QuantumCircuit(n, n)
    qc.append(ansatz.to_gate(label='U'), qc.qubits)
    qc.measure(range(n), range(n))
    return density_from_counts(ex.counts(qc, shots), n, shots)


def expectation(method, ansatz, problem, ex, shots):
    """Compute expectation val for given decomposition method and ansatz;
    ux if a potential is present"""
    val = 0.0
    for start, m, t in problem.dims:
        val += t * DIR_VALUE[method](ansatz, problem.n, start, m, ex, shots)
    if problem.ux is not None:
        val += float(density(ansatz, problem.n, ex, shots) @ problem.ux)
    return val


def n_circuits(method, problem):
    per_dir = {'SD': lambda m: m, 'FD': lambda m: 2, 'PO': lambda m: 2}[method]
    total = sum(per_dir(m) for _, m, _ in problem.dims)
    if problem.ux is not None:
        total += 1
    return total
