"""Parameter-shift gradients + Adam. The shift is finite, so shot noise enters the gradient
additively instead of divided by a small step, which is why finite differences fail here."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from .ansatz import make_HEA, QFT_LNN, increment
from .estimators import _p, _ixZ, _i0xZ, density_from_counts

# ((steps, shots_per_circuit), ...) — coarse first, then refine. The learning rate is not
# scheduled.
DEFAULT_STAGES = ((200, 2**18), (100, 2**20))
DEFAULT_LR = 0.05


class BatchedExpectation:
    """<H> evaluator and parameter-shift gradient for one (problem, method, depth) triple.

    Templates are transpiled once, so an evaluation is a single run_counts() batch."""

    def __init__(self, problem, method, depth, ex):
        self.problem, self.method, self.depth, self.ex = problem, method, depth, ex
        self.P = problem.n * depth
        self.pv = ParameterVector('t', self.P)
        hea = make_HEA(problem.n, depth, self.pv)
        self.blocks = []          # (n_circuits, decode(counts_slice, shots) -> value)
        circs = []
        for start, m, t in problem.dims:
            c = self._dir_circuits(hea, problem.n, start, m)
            circs += c
            self.blocks.append((len(c), self._dir_decoder(start, m, t)))
        if problem.ux is not None:
            qc = QuantumCircuit(problem.n, problem.n)
            qc.append(hea.to_gate(label='U'), qc.qubits)
            qc.measure(range(problem.n), range(problem.n))
            circs.append(qc)
            ux = problem.ux
            self.blocks.append((1, lambda cs, shots, ux=ux, n=problem.n:
                                float(density_from_counts(cs[0], n, shots) @ ux)))
        self.templates = [ex.transpile(c) for c in circs]
        self.per_eval = len(self.templates)

    def _dir_circuits(self, hea, n, start, m):
        u = hea.to_gate(label='U')
        if self.method == 'SD':
            out = []
            for j in range(m):
                mm = m - j
                qc = QuantumCircuit(n, mm)
                qc.append(u, qc.qubits)
                for i in range(m - j - 1):
                    qc.cx(start + i + 1, start + i)
                qc.h(start + m - 1 - j)
                qc.measure(range(start, start + mm), range(mm))
                out.append(qc)
            return out
        if self.method == 'FD':
            qc = QuantumCircuit(n + 1, 1)
            qc.append(u, qc.qubits[:n])
            QFT_LNN(qc, m, start); qc.h(n)
            for idx in range(m):
                qc.cp((2 * np.pi / (2**m)) * (2**idx), n, start + m - 1 - idx)
            qc.h(n); qc.measure(n, 0)
            qc2 = QuantumCircuit(n, m)
            qc2.append(u, qc2.qubits)
            increment(qc2, m, start); qc2.h(start)
            qc2.measure(range(start, start + m), range(m))
            return [qc, qc2]
        if self.method == 'PO':
            c1 = QuantumCircuit(n, 1)
            c1.append(u, c1.qubits)
            c1.h(start); c1.measure(start, 0)
            c2 = QuantumCircuit(n, m)
            c2.append(u, c2.qubits)
            increment(c2, m, start); c2.h(start)
            c2.measure(range(start, start + m), range(m))
            return [c1, c2]
        raise ValueError(f"unknown method {self.method}")

    def _dir_decoder(self, start, m, t):
        if self.method == 'SD':
            def dec(cs, shots):
                val = 0.0
                for j, cnt in enumerate(cs):
                    mm = m - j
                    if j == m - 1:
                        val += _p(cnt, '0', shots) - _p(cnt, '1', shots)
                    else:
                        tail = '0' * (mm - 2)
                        val += _p(cnt, tail + '10', shots) - _p(cnt, tail + '11', shots)
                return t * (2 - val)
        elif self.method == 'FD':
            def dec(cs, shots):
                A_per = 2 * (cs[0].get('0', 0) - cs[0].get('1', 0)) / shots - 2
                return t * (-A_per + _i0xZ(cs[1], shots, m))
        else:                                    # PO
            def dec(cs, shots):
                return t * (2 - _ixZ(cs[0], shots) - _ixZ(cs[1], shots)
                            + _i0xZ(cs[1], shots, m))
        return dec

    # -------- evaluation --------
    def _bind(self, theta):
        b = dict(zip(self.pv, theta))
        return [tc.assign_parameters(b) for tc in self.templates]

    def enable_readout_mitigation(self, cal_shots=2**20):
        """Correct every counts dict by the tensored inverse of the readout confusion matrix."""
        from .mitigation import ReadoutMitigator, measure_map
        n_phys = max(t.num_qubits for t in self.templates)
        mit = ReadoutMitigator(self.ex, n_phys, shots=cal_shots)
        self._correctors = [mit.corrector(measure_map(t)) for t in self.templates]
        self.mitigator = mit
        return mit

    def enable_zne(self, factors=(1, 3, 5)):
        """Fold the 2q gates at each factor and Richardson-extrapolate to zero noise.
        Costs len(factors)x circuits per evaluation."""
        from .mitigation import fold_2q, richardson_coeffs
        self.zne_factors = tuple(factors)
        self.zne_coeffs = richardson_coeffs(self.zne_factors)
        self._folded = {c: [fold_2q(t, c) for t in self.templates]
                        for c in self.zne_factors}
        return self.zne_coeffs

    def _decode(self, counts, shots):
        if getattr(self, '_correctors', None):
            counts = [fix(c) for fix, c in zip(self._correctors, counts)]
        val, i = 0.0, 0
        for k, dec in self.blocks:
            val += dec(counts[i:i + k], shots)
            i += k
        return val

    def _values(self, thetas, shots):
        """<M> for a list of parameter vectors in one batched run, ZNE-extrapolated if on."""
        factors = getattr(self, 'zne_factors', None) or (1,)
        templ = {1: self.templates} if factors == (1,) else self._folded
        batch = []
        for th in thetas:
            b = dict(zip(self.pv, th))
            for c in factors:
                batch += [tc.assign_parameters(b) for tc in templ[c]]
        counts = self.ex.run_counts(batch, shots)
        k, idx, vals = self.per_eval, 0, []
        for th in thetas:
            per_factor = []
            for c in factors:
                per_factor.append(self._decode(counts[idx:idx + k], shots))
                idx += k
            if len(factors) == 1:
                vals.append(per_factor[0])
            else:
                vals.append(float(np.dot(self.zne_coeffs, per_factor)))
        return np.array(vals)

    def value(self, theta, shots):
        return float(self._values([np.asarray(theta, float)], shots)[0])

    def psgrad(self, theta, shots):
        """Parameter-shift gradient: ALL 2P shifted evaluations in ONE batched run."""
        thetas = []
        for i in range(self.P):
            tp = theta.copy(); tp[i] += np.pi / 2
            tm = theta.copy(); tm[i] -= np.pi / 2
            thetas += [tp, tm]
        v = self._values(thetas, shots)
        return 0.5 * (v[0::2] - v[1::2])


def exact_psgrad(theta, problem, depth):
    """Parameter-shift on the exact expectation: the reference `exact_grad` is tested against."""
    from qiskit.quantum_info import Statevector

    def cost(th):
        psi = np.real(Statevector(make_HEA(problem.n, depth, th)).data)
        return float(psi @ problem.M @ psi)
    g = np.zeros(len(theta))
    for i in range(len(theta)):
        tp = theta.copy(); tp[i] += np.pi / 2
        tm = theta.copy(); tm[i] -= np.pi / 2
        g[i] = 0.5 * (cost(tp) - cost(tm))
    return g


# Noise-free path used for basin screening and references: bare-numpy HEA simulator +
# adjoint gradient. RY+CNOT keeps every amplitude real and every gate orthogonal, so one
# forward pass and one backward sweep give all P derivatives instead of 2P simulations.
def _ry_(st, n, q, theta):
    """In-place RY(theta) on qubit q (Qiskit order: qubit 0 = LSB of the state index)."""
    v = st.reshape(2 ** (n - 1 - q), 2, 2 ** q)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    a0 = v[:, 0, :].copy()
    v[:, 0, :] = c * a0 - s * v[:, 1, :]
    v[:, 1, :] = s * a0 + c * v[:, 1, :]
    return st


def _cx_(st, n, ctrl, targ):
    """In-place CNOT (self-inverse, so it doubles as its own adjoint)."""
    lo, hi = min(ctrl, targ), max(ctrl, targ)
    v = st.reshape(2 ** (n - 1 - hi), 2, 2 ** (hi - lo - 1), 2, 2 ** lo)
    if ctrl == hi:                      # control is the high qubit -> flip the low one
        tmp = v[:, 1, :, 0, :].copy()
        v[:, 1, :, 0, :] = v[:, 1, :, 1, :]
        v[:, 1, :, 1, :] = tmp
    else:                               # control is the low qubit -> flip the high one
        tmp = v[:, 0, :, 1, :].copy()
        v[:, 0, :, 1, :] = v[:, 1, :, 1, :]
        v[:, 1, :, 1, :] = tmp
    return st


def _ytilde(st, n, q):
    """Real generator of RY: d/dtheta RY = RY @ (1/2)*[[0,-1],[1,0]]. Returns a new array."""
    out = np.empty_like(st)
    v = st.reshape(2 ** (n - 1 - q), 2, 2 ** q)
    o = out.reshape(2 ** (n - 1 - q), 2, 2 ** q)
    o[:, 0, :] = -v[:, 1, :]
    o[:, 1, :] = v[:, 0, :]
    return out


def statevector(theta, n, depth):
    """Real statevector of make_HEA(n, depth, theta) — identical to Qiskit's, ~100x cheaper."""
    st = np.zeros(2 ** n)
    st[0] = 1.0
    for d in range(depth):
        for i in range(n):
            _ry_(st, n, i, theta[i + d * n])
        for i in range(n - 1):
            _cx_(st, n, i, i + 1)
    return st


def exact_cost(theta, problem, depth):
    psi = statevector(theta, problem.n, depth)
    return float(psi @ problem.M @ psi)


def exact_grad(theta, problem, depth):
    """Adjoint gradient of <psi(theta)|M|psi(theta)>, identical to exact_psgrad but ~P times
    cheaper."""
    n = problem.n
    phi = statevector(theta, n, depth)          # phi = |psi>
    lam = problem.M @ phi                       # lambda_L = M|psi>
    g = np.zeros(len(theta))
    for d in reversed(range(depth)):
        for i in reversed(range(n - 1)):        # undo the CNOT chain (self-inverse)
            _cx_(phi, n, i, i + 1)
            _cx_(lam, n, i, i + 1)
        for i in reversed(range(n)):            # phi is the state AFTER this RY
            j = i + d * n
            g[j] = lam @ _ytilde(phi, n, i)
            _ry_(phi, n, i, -theta[j])
            _ry_(lam, n, i, -theta[j])
    return g


class _ResumableADAM:
    """qiskit_algorithms.ADAM's update loop with (t, m, v) injectable, so a resumed run keeps
    its moments instead of taking an lr*sign(g) first step."""

    def __init__(self, maxiter, lr, beta_1, beta_2, noise_factor):
        self._maxiter, self._lr = maxiter, lr
        self._beta_1, self._beta_2, self._noise_factor = beta_1, beta_2, noise_factor
        self._t = 0
        self._m = self._v = None

    def minimize(self, x0, jac, init_state=None):
        params = params_new = np.asarray(x0, float)
        if init_state is None:
            derivative = jac(params)
            self._t = 0
            self._m = np.zeros(np.shape(derivative))
            self._v = np.zeros(np.shape(derivative))
        else:
            self._t, self._m, self._v = init_state
            derivative = None                      # recomputed at loop top (t > 0)
        while self._t < self._maxiter:
            if self._t > 0:
                derivative = jac(params)
            self._t += 1
            self._m = self._beta_1 * self._m + (1 - self._beta_1) * derivative
            self._v = self._beta_2 * self._v + (1 - self._beta_2) * derivative * derivative
            lr_eff = (self._lr * np.sqrt(1 - self._beta_2 ** self._t)
                      / (1 - self._beta_1 ** self._t))
            params_new = params - lr_eff * self._m.flatten() / (
                np.sqrt(self._v.flatten()) + self._noise_factor)
            params = params_new
        return params_new


def adam(grad_fn, x0, stages=DEFAULT_STAGES, callback=None, lr=DEFAULT_LR,
         beta_1=0.9, beta_2=0.99, eps=1e-8, state_file=None, resume=False):
    """Adam over measured gradients, grad_fn(theta, shots) -> gradient, returning the final
    parameters. The whole shot schedule runs in one minimize() call so the moments survive
    the stage change; state_file snapshots (t, m, v, x) for resume=True."""
    import os as _os

    if stages and len(stages[0]) != 2:
        raise ValueError("stages must be ((steps, shots), ...); the learning rate is now the "
                         "separate `lr` argument (it is no longer scheduled)")
    shots_at = [s for steps, s in stages for _ in range(steps)]
    maxiter = len(shots_at)
    x_start, init_state, start_k = np.asarray(x0, float), None, 0
    if resume and state_file and _os.path.exists(state_file):
        z = np.load(state_file)
        init_state = (int(z['t']), z['m'], z['v'])
        x_start, start_k = z['x'], int(z['t'])
    state = {'k': start_k}
    opt = _ResumableADAM(maxiter=maxiter, lr=lr, beta_1=beta_1, beta_2=beta_2,
                         noise_factor=eps)

    def jac(theta):
        th = np.asarray(theta, float)
        k = min(state['k'], maxiter - 1)
        state['k'] += 1
        if callback:
            callback(th, k)
        if state_file:                              # snapshot before this step's update
            np.savez(state_file + '.tmp', t=opt._t,
                     m=opt._m if opt._m is not None else np.zeros_like(th),
                     v=opt._v if opt._v is not None else np.zeros_like(th), x=th)
            _os.replace(state_file + '.tmp.npz', state_file)
        return grad_fn(th, shots_at[k])

    return np.asarray(opt.minimize(x_start, jac, init_state=init_state), float)
