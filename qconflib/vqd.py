"""First excited states by deflation: minimise <H> + beta |<psi0|psi>|^2, with the overlap
measured by a compute-uncompute circuit so the method stays hardware-executable."""
import numpy as np
from qiskit import QuantumCircuit
from .ansatz import make_HEA
from .backends import Executor
from .estimators import density
from .gradients import (BatchedExpectation, adam, exact_grad, statevector,
                        DEFAULT_STAGES, DEFAULT_LR)


def _shim(problem, Mdef):
    """Problem stand-in carrying the deflated matrix, for the exact-gradient path."""
    from types import SimpleNamespace
    return SimpleNamespace(n=problem.n, M=Mdef)


def default_beta(problem):
    """Must exceed E1-E0; the floor keeps beta useful when a strong potential closes the gap."""
    return max(0.5, 4.0 * problem.gap01)


def deflated_matrix(problem, psi0, beta):
    """H + beta |psi0><psi0|, with psi0 the TRAINED ground state. Used for screening."""
    return problem.M + beta * np.outer(psi0, psi0)


def add_deflation_block(bx, theta0, beta, ex):
    """Append the overlap block to `bx` in place; afterwards it evaluates the deflated cost."""
    n, depth = bx.problem.n, bx.depth
    qc = QuantumCircuit(n, n)
    qc.compose(make_HEA(n, depth, bx.pv), inplace=True)
    qc.compose(make_HEA(n, depth, theta0).inverse(), inplace=True)
    qc.measure(range(n), range(n))
    bx.templates.append(ex.transpile(qc))
    key = '0' * n
    bx.blocks.append((1, lambda cs, shots, b=beta, k=key: b * cs[0].get(k, 0) / shots))
    bx.per_eval = len(bx.templates)
    return bx


def subspace_fidelity(psi, problem, level=1, atol=1e-9):
    """|P_deg psi|^2 against the (possibly degenerate) eigenspace of `level`."""
    w = problem.eigvals
    cols = np.where(np.isclose(w, w[level], atol=atol))[0]
    V = problem.eigvecs[:, cols]
    return float(np.linalg.norm(V.T @ psi) ** 2), len(cols)


def screen_excited_init(problem, Mdef, depth, seeds=range(16), keep=1, short=300,
                        refine=800, ref=None, thresh=0.02):
    """Return the raw init whose classical deflated descent lands lowest, plus its info dict.

    keep>1 re-descends the best `keep` seeds, needed where a short descent ranks unreliably."""
    shim = _shim(problem, Mdef)

    def descend(x0, steps):
        return adam(lambda th, _: exact_grad(th, shim, depth), x0,
                    stages=((steps, 0),), lr=DEFAULT_LR)

    def score(x):
        psi = statevector(x, problem.n, depth)
        if ref is not None:
            return np.sqrt(max(0.0, 1 - abs(np.vdot(psi, ref)) ** 2))
        return float(psi @ Mdef @ psi)

    scored = []
    for s in seeds:
        x0 = np.random.default_rng(s).uniform(-np.pi, np.pi, problem.n * depth)
        v = score(descend(x0, short))
        scored.append((v, s, x0))
        if keep == 1 and ref is not None and v < thresh:
            break
    scored.sort(key=lambda t: t[0])
    if keep == 1:
        v, s, x0 = scored[0]
        return x0, {'seed': int(s), 'score': float(v), 'screened': len(scored)}
    best = None
    for _, s, x0 in scored[:keep]:
        v = score(descend(x0, refine))
        if best is None or v < best[0]:
            best = (v, s, x0)
    return best[2], {'seed': int(best[1]), 'score': float(best[0]),
                     'screened': len(scored), 'refined': keep}


def solve_excited(problem, theta0, method='SD', depth=3, beta=None, shots=2**17,
                  device='auto', noise_from=None, seed=None, stages=DEFAULT_STAGES,
                  lr=DEFAULT_LR, x0=None, screen=16, keep=1, screen_ref=None,
                  mitigate=None, ex=None):
    """First excited state, deflated against the TRAINED ground state `theta0`.

    Returns a dict with x_opt, energy, subspace fidelity, degeneracy, trace and rho."""
    if beta is None:
        beta = default_beta(problem)
    psi0 = statevector(theta0, problem.n, depth)
    Mdef = deflated_matrix(problem, psi0, beta)
    info = {}
    if x0 is None:
        x0, info = screen_excited_init(problem, Mdef, depth, seeds=range(screen),
                                       keep=keep, ref=screen_ref)

    ex = ex or Executor(device, noise_from, seed=seed)
    if method == 'classical':
        defl = _shim(problem, Mdef)
        grad = lambda th, _: exact_grad(th, defl, depth)
    else:
        bx = BatchedExpectation(problem, method, depth, ex)
        if mitigate and 'readout' in mitigate:
            bx.enable_readout_mitigation()
        if mitigate and 'zne' in mitigate:
            bx.enable_zne()
        add_deflation_block(bx, theta0, beta, ex)
        grad = bx.psgrad

    hist = []

    def cb(th, _i):
        fid, _ = subspace_fidelity(statevector(th, problem.n, depth), problem)
        hist.append(np.sqrt(max(0.0, 1 - fid)))

    x_opt = adam(grad, x0, stages=stages, lr=lr, callback=cb)
    psi = statevector(x_opt, problem.n, depth)
    fid, degen = subspace_fidelity(psi, problem)
    rho = density(make_HEA(problem.n, depth, x_opt), problem.n,
                  ex if method != 'classical' else Executor(device), shots)
    return {'history': hist, 'x_opt': x_opt, 'energy': float(psi @ problem.M @ psi),
            'E1_exact': float(problem.eigvals[1]), 'fidelity': fid, 'degeneracy': degen,
            'trace': float(np.sqrt(max(0.0, 1 - fid))),
            'overlap0': float(abs(np.vdot(psi, psi0)) ** 2), 'rho': rho, 'beta': float(beta),
            'init': info, 'method': method}
