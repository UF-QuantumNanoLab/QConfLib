"""Error mitigation applied inside the training loop, per cost and gradient evaluation,
because gate noise biases the landscape the optimiser descends and not just its final value."""
import numpy as np
from qiskit import QuantumCircuit

FOLDABLE_2Q = ('ecr', 'cx', 'cz')   # self-inverse entangling gates: G^c = G for odd c


def fold_2q(tqc, c):
    """Repeat each self-inverse entangling gate c times: same unitary, c-fold gate noise.

    Runs a TRANSPILED circuit and is never re-transpiled, so nothing can cancel the folds."""
    if c == 1:
        return tqc
    assert c % 2 == 1, "fold factor must be odd (G^c = G needs odd c)"
    out = tqc.copy_empty_like()
    for inst in tqc.data:
        reps = c if inst.operation.name in FOLDABLE_2Q else 1
        for _ in range(reps):
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def richardson_coeffs(factors):
    """Lagrange coefficients extrapolating to zero noise. For (1,3,5): (15/8, -5/4, 3/8)."""
    lam = np.asarray(factors, float)
    coef = np.array([np.prod([lj / (lj - li) for lj in lam if lj != li])
                     for li in lam])
    assert abs(coef.sum() - 1) < 1e-12
    return coef


def measure_map(tqc):
    """clbit index -> physical qubit index, read off a transpiled circuit's measures."""
    m = {}
    for inst in tqc.data:
        if inst.operation.name == 'measure':
            q = tqc.find_bit(inst.qubits[0]).index
            c = tqc.find_bit(inst.clbits[0]).index
            m[c] = q
    return [m[c] for c in range(len(m))]


class ReadoutMitigator:
    """Per-qubit readout calibration (2 circuits) + tensored-inverse counts correction.

    The clbit -> physical map is read off each transpiled circuit, since routing permutes it."""

    def __init__(self, ex, n_qubits, shots=2**20):
        self.n = n_qubits
        cal0 = QuantumCircuit(n_qubits, n_qubits)
        cal1 = QuantumCircuit(n_qubits, n_qubits)
        cal1.x(range(n_qubits))
        for qc in (cal0, cal1):
            qc.measure(range(n_qubits), range(n_qubits))
        t0, t1 = ex.transpile(cal0), ex.transpile(cal1)
        c0, c1 = ex.run_counts([t0, t1], shots)
        map0, map1 = measure_map(t0), measure_map(t1)
        e0 = np.zeros(n_qubits)   # P(read 1 | prep 0), per physical qubit
        e1 = np.zeros(n_qubits)   # P(read 0 | prep 1)
        for c in range(n_qubits):
            e0[map0[c]] = sum(v for k, v in c0.items() if k[-1 - c] == '1') / shots
            e1[map1[c]] = sum(v for k, v in c1.items() if k[-1 - c] == '0') / shots
        self.e0, self.e1 = e0, e1
        self._inv2 = [np.linalg.inv(np.array([[1 - e0[q], e1[q]], [e0[q], 1 - e1[q]]]))
                      for q in range(n_qubits)]

    def corrector(self, clbit_phys):
        """counts -> corrected quasi-counts for one template. kron order: clbit m-1 first."""
        m = len(clbit_phys)
        A = np.array([[1.0]])
        for c in range(m - 1, -1, -1):
            A = np.kron(A, self._inv2[clbit_phys[c]])
        keys = [format(i, f'0{m}b') for i in range(2 ** m)]

        def correct(counts):
            p = np.zeros(2 ** m)
            for k, v in counts.items():
                p[int(k, 2)] = v
            q = A @ p
            return {key: q[i] for i, key in enumerate(keys) if q[i] != 0.0}
        return correct
