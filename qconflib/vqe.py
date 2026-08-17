"""Ground-state VQE: classically screened init + parameter-shift Adam, or any scipy method."""
import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import Statevector
from .ansatz import make_HEA
from .backends import Executor
from .estimators import expectation, density
from .gradients import (BatchedExpectation, exact_grad, adam, DEFAULT_STAGES,
                        DEFAULT_LR, statevector as _fast_sv)


def _metrics(theta, problem, depth):
    psi = _fast_sv(theta, problem.n, depth)
    cost = float(psi @ problem.M @ psi)
    ov = abs(np.vdot(psi, problem.ground))
    trace = np.sqrt(max(0.0, 1 - ov ** 2))
    L2 = float(np.linalg.norm(np.abs(psi) - np.abs(problem.ground)))
    return cost, L2, trace


def screen_init(problem, depth, seeds=range(16), thresh=0.2, optimizer='Adam',
                maxiter=400):
    """Return the best raw init, scored by descending the exact (noiseless) problem.

    Only a minority of raw seeds reach the ground basin, and the noisy gradient tracks the
    exact one closely enough that the basin can be predicted classically."""
    n = problem.n

    def trace_of(theta):
        return _metrics(theta, problem, depth)[2]

    best = None
    for s in seeds:
        x0 = np.random.default_rng(s).uniform(-np.pi, np.pi, n * depth)
        if optimizer == 'Adam':
            x = adam(lambda th, shots: exact_grad(th, problem, depth), x0,
                     stages=tuple((st, 0) for st, _ in DEFAULT_STAGES))
        else:
            def cost(theta):
                psi = _fast_sv(theta, n, depth)
                return float(psi @ problem.M @ psi)
            x = minimize(cost, x0, method=optimizer,
                         options={'maxiter': maxiter, 'tol': 1e-5}).x
        tr = trace_of(x)
        if best is None or tr < best[0]:
            best = (tr, x0, s)
        if tr < thresh:
            break
    return best[1], {'seed': best[2], 'classical_trace': best[0]}


def solve(problem, method='SD', depth=3, shots=2**17, maxiter=300, optimizer='Adam',
          device='auto', noise_from=None, x0=None, screen=16, seed=None,
          stages=DEFAULT_STAGES, lr=DEFAULT_LR, ckpt=None, mitigate=None,
          resume=False, parallel='auto'):
    """Ground state for `problem`. method: 'classical' (exact cost) or 'PO'/'SD'/'FD' (shots).

    stages is the coarse-to-fine shot schedule for Adam; shots sets the density readout.
    mitigate takes 'readout' and/or 'zne'; ckpt/resume make a long run restartable.

    Returns a dict with history, x_opt, terminal energy/L2/trace, rho and lam_min."""
    ex = None if method == 'classical' else Executor(device, noise_from, seed=seed,
                                                     parallel_experiments=parallel)
    if x0 is None:
        x0, _ = screen_init(problem, depth, range(screen),
                            optimizer=optimizer if optimizer != 'Adam' else 'Adam')
    hist = []
    state_file = f"{ckpt['prefix']}_adamstate.npz" if ckpt else None
    if resume and ckpt:
        import os as _os
        hfile = f"{ckpt['prefix']}_history.csv"
        if _os.path.exists(hfile) and _os.path.exists(state_file):
            import numpy as _np
            prev = _np.loadtxt(hfile, delimiter=',', skiprows=1)
            t0 = int(_np.load(state_file)['t'])
            hist = [tuple(r) for r in _np.atleast_2d(prev)[:t0]]  # align with resume step
            print(f"[QConfLib] resuming from step {t0} (moments restored, "
                  f"{len(hist)} history rows kept)", flush=True)

    def _flush_ckpt(theta):
        import os as _os
        _os.makedirs(_os.path.dirname(ckpt['prefix']) or '.', exist_ok=True)
        np.savetxt(f"{ckpt['prefix']}_history.csv", np.array(hist), delimiter=',',
                   header='cost,L2,trace', comments='')
        np.savetxt(f"{ckpt['prefix']}_params.csv", theta, delimiter=',')

    if optimizer == 'Adam':
        if method == 'classical':
            grad = lambda th, s: exact_grad(th, problem, depth)
        else:
            bx = BatchedExpectation(problem, method, depth, ex)
            if mitigate and 'readout' in mitigate:
                mit = bx.enable_readout_mitigation()
                print(f"[QConfLib] readout mitigation on: e0={np.round(mit.e0, 4)} "
                      f"e1={np.round(mit.e1, 4)}", flush=True)
            if mitigate and 'zne' in mitigate:
                coeffs = bx.enable_zne()
                print(f"[QConfLib] ZNE on: fold factors {bx.zne_factors}, Richardson "
                      f"coeffs {np.round(coeffs, 4)} ({len(bx.zne_factors)}x circuits)",
                      flush=True)
            grad = bx.psgrad

        def _cb(th, k):
            hist.append(_metrics(th, problem, depth))
            if ckpt and (k + 1) % ckpt['every'] == 0:
                _flush_ckpt(th)
        x_opt = adam(grad, x0, stages=stages, lr=lr, callback=_cb,
                     state_file=state_file, resume=resume)
        if ckpt:
            _flush_ckpt(x_opt)
    else:
        def objective(theta):
            if method == 'classical':
                psi = _fast_sv(theta, problem.n, depth)
                c = float(psi @ problem.M @ psi)
            else:
                c = expectation(method, make_HEA(problem.n, depth, theta), problem, ex,
                                shots)
            hist.append(_metrics(theta, problem, depth))
            return c
        x_opt = minimize(objective, x0, method=optimizer,
                         options={'maxiter': maxiter, 'tol': 1e-4}).x

    cost, L2, trace = _metrics(x_opt, problem, depth)
    rho_ex = Executor(device) if method == 'classical' else ex
    if mitigate and method != 'classical' and optimizer == 'Adam':
        # the final density readout goes through the same noisy measurement — mitigate it
        # with the same layers as the training loop (readout correction / per-bin ZNE)
        from qiskit import QuantumCircuit
        from .estimators import density_from_counts
        from .mitigation import measure_map, fold_2q, richardson_coeffs
        qc = QuantumCircuit(problem.n, problem.n)
        qc.append(make_HEA(problem.n, depth, x_opt).to_gate(label='U'), qc.qubits)
        qc.measure(range(problem.n), range(problem.n))
        tqc = ex.transpile(qc)
        factors = bx.zne_factors if 'zne' in mitigate else (1,)
        fix = (bx.mitigator.corrector(measure_map(tqc)) if 'readout' in mitigate
               else (lambda c: c))
        counts = ex.run_counts([fold_2q(tqc, c) for c in factors], shots)
        rhos = [density_from_counts(fix(c), problem.n, shots) for c in counts]
        rho = (rhos[0] if len(factors) == 1
               else np.maximum(0.0, richardson_coeffs(factors) @ np.array(rhos)))
    else:
        rho = density(make_HEA(problem.n, depth, x_opt), problem.n, rho_ex, shots)
    return {'history': hist, 'x_opt': x_opt, 'energy': cost, 'L2': L2, 'trace': trace,
            'rho': rho, 'lam_min': problem.lam_min, 'evals': len(hist), 'method': method,
            'optimizer': optimizer}
