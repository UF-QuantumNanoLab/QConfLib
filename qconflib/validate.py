import numpy as np
from qiskit.quantum_info import Statevector
from .ansatz import make_HEA
from .backends import Executor
from .estimators import DIR_VALUE, density, expectation
from .operators import _embed, box_kinetic


def validate_problem(problem, depth=3, shots=2_000_000, device='auto', seed=11, tol=None,
                     verbose=True):
    # used to validate the estimators against exact expectations for a random ansatz state
    n = problem.n
    ex = Executor(device)
    rng = np.random.default_rng(seed)
    ansatz = make_HEA(n, depth, rng.uniform(0, 2 * np.pi, n * depth))
    psi = np.real(Statevector(ansatz).data)
    s = 1.0 / np.sqrt(shots)
    tol_dir = tol if tol is not None else 4 * 2 * s
    t_norm = np.sqrt(sum(t**2 for _, _, t in problem.dims))
    tol_H = tol if tol is not None else 4 * 2 * t_norm * s
    ok = True
    for start, m, t in problem.dims:
        Md = _embed(box_kinetic(2**m), start, m, n)
        exact = float(psi @ Md @ psi)
        for name, est in DIR_VALUE.items():  # loop over the three estimators (SD / FD / PO)
            v = est(ansatz, n, start, m, ex, shots)
            d = abs(v - exact)
            ok &= d < tol_dir
            if verbose:
                print(f"  dir[{start}:{start + m}] {name}: {v:+.4f} vs exact {exact:+.4f} "
                      f"(d={d:.4f}, tol={tol_dir:.4f}) [{'OK' if d < tol_dir else 'FAIL'}]")
    if problem.ux is not None:
        exact_u = float((np.abs(psi)**2) @ problem.ux)
        vu = float(density(ansatz, n, ex, shots) @ problem.ux)
        tol_u = tol if tol is not None else 4 * float(np.max(np.abs(problem.ux))) * s
        d = abs(vu - exact_u)
        ok &= d < tol_u
        if verbose:
            print(f"  diag(u): {vu:+.4f} vs exact {exact_u:+.4f} (d={d:.4f}, tol={tol_u:.4f}) "
                  f"[{'OK' if d < tol_u else 'FAIL'}]")
    exact_H = float(psi @ problem.M @ psi)
    for name in DIR_VALUE:
        vH = expectation(name, ansatz, problem, ex, shots)
        d = abs(vH - exact_H)
        ok &= d < tol_H
        if verbose:
            print(f"  <H> via {name}: {vH:+.4f} vs exact {exact_H:+.4f} "
                  f"(d={d:.4f}, tol={tol_H:.4f}) [{'OK' if d < tol_H else 'FAIL'}]")
    return ok
