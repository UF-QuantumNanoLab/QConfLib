#!/usr/bin/env python
"""QConfLib CLI — solve a confinement ground state with one or more methods.

Examples:
  python cli/solve.py --dim 1 --n 5 --method SD
  python cli/solve.py --dim 1 --n 5 --u0 1.0 --method classical,SD,FD --maxiter 300
  python cli/solve.py --dim 2 --nx 3 --ny 3 --depth 4 --method SD,FD
  python cli/solve.py --dim 1 --n 4 --method SD --noise-from sherbrooke --shots 20000
"""
import argparse
import os
import time
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qconflib import problem_1d, problem_2d, problem_3d, screen_init, solve  # noqa: E402
from qconflib.gradients import DEFAULT_STAGES, DEFAULT_LR  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="QConfLib ground-state solver")
    ap.add_argument('--dim', type=int, default=1, choices=[1, 2, 3])
    ap.add_argument('--n', type=int, default=5, help="[1D] number of qubits")
    ap.add_argument('--u0', type=float, default=0.0, help="[1D] potential strength in eV")
    ap.add_argument('--nx', type=int, default=3, help="[2D] x qubits (low register)")
    ap.add_argument('--ny', type=int, default=3, help="[2D] y qubits (high register)")
    ap.add_argument('--t1', type=float, default=1 / 0.19, help="[2D] TB weight along x")
    ap.add_argument('--t2', type=float, default=1 / 0.98, help="[2D] TB weight along y")
    ap.add_argument('--nz', type=int, default=3, help="[3D] z qubits (high register)")
    ap.add_argument('--t3', type=float, default=1.0, help="[3D] TB weight along z")
    ap.add_argument('--method', default='classical,SD,FD',
                    help="comma list from classical/PO/SD/FD")
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--shots', type=int, default=2**17, help="shots per circuit")
    ap.add_argument('--maxiter', type=int, default=300)
    ap.add_argument('--optimizer', default='Adam',
                    help="'Adam' (parameter-shift gradients, recommended) or any "
                         "scipy.optimize.minimize method (COBYLA kept for comparison)")
    ap.add_argument('--seed', type=int, default=None,
                    help="frozen simulator seed (CRN) -> fully reproducible noisy runs")
    ap.add_argument('--stages', default=None,
                    help="[Adam] shots schedule steps:shots,... (default 200:262144,100:1048576)")
    ap.add_argument('--lr', type=float, default=None, help="[Adam] learning rate (default 0.05)")
    ap.add_argument('--device', default='auto', choices=['auto', 'GPU', 'CPU'])
    ap.add_argument('--noise-from', default=None,
                    help="IBM calibration snapshot (e.g. sherbrooke/torino); omit = ideal")
    ap.add_argument('--screen', type=int, default=16, help="random inits screened classically")
    ap.add_argument('--init-seed', type=int, default=None,
                    help="use this raw init seed directly (skip screening) — e.g. a basin "
                         "already known good from a classical scan")
    ap.add_argument('--init-file', default=None,
                    help="load theta_0 from a .npy/.csv file (skip screening) — e.g. a "
                         "physics-informed init fitted to the analytic free-box sine")
    ap.add_argument('--mitigate', default=None,
                    help="comma list of in-loop error-mitigation layers (e.g. 'readout' or "
                         "'readout,zne'); applied to every evaluation AND the final density")
    ap.add_argument('--resume', action='store_true',
                    help="continue a crashed Adam run from the ckpt (t,m,v,x) snapshot — "
                         "moments restored, no restart kick")
    ap.add_argument('--parallel', default='auto',
                    help="Aer max_parallel_experiments (int, or 'auto' = cores capped 24); "
                         "lower to reduce thermal load")
    ap.add_argument('--out', default='outputs/run', help="output prefix (png/csv)")
    a = ap.parse_args()

    problem = (problem_1d(a.n, a.u0) if a.dim == 1 else
               problem_2d(a.nx, a.ny, a.t1, a.t2) if a.dim == 2 else
               problem_3d(a.nx, a.ny, a.nz, a.t1, a.t2, a.t3))
    methods = [m.strip() for m in a.method.split(',')]
    stages = DEFAULT_STAGES if a.stages is None else tuple(
        tuple(int(v) for v in s.split(':')) for s in a.stages.split(','))
    lr = DEFAULT_LR if a.lr is None else a.lr
    print(f"[QConfLib] {problem.label}  ({problem.n} qubits)  lam_min={problem.lam_min:.5f}")
    print(f"[QConfLib] depth={a.depth} optimizer={a.optimizer} lr={lr} stages={stages} "
          f"density-shots={a.shots} "
          f"backend={'ideal ' + a.device if not a.noise_from else a.noise_from}", flush=True)

    if a.init_file is not None:
        x0 = (np.load(a.init_file) if a.init_file.endswith('.npy')
              else np.loadtxt(a.init_file, delimiter=','))
        x0 = np.asarray(x0, float).ravel()
        if x0.size != problem.n * a.depth:
            raise SystemExit(f"--init-file has {x0.size} params, expected "
                             f"{problem.n * a.depth} (n*depth)")
        info = {'init_file': a.init_file, 'classical_trace': None, 'screened': False}
        print(f"[QConfLib] init: file={a.init_file} (given, screening skipped)")
    elif a.init_seed is not None:
        x0 = np.random.default_rng(a.init_seed).uniform(
            -np.pi, np.pi, problem.n * a.depth)
        info = {'seed': a.init_seed, 'classical_trace': None, 'screened': False}
        print(f"[QConfLib] init: seed={a.init_seed} (given, screening skipped)")
    else:
        x0, info = screen_init(problem, a.depth, range(a.screen), optimizer=a.optimizer)
        info['screened'] = True
        print(f"[QConfLib] screened init: seed={info['seed']} "
              f"classical_trace={info['classical_trace']:.3f}")

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    results, times = {}, {}
    for m in methods:
        t0 = time.perf_counter()
        # ckpt: flush partial history+params every 25 steps (survives mid-method kills)
        ckdir = f"{a.out}_ckpt"
        os.makedirs(ckdir, exist_ok=True)
        r = solve(problem, method=m, depth=a.depth, shots=a.shots, maxiter=a.maxiter,
                  optimizer=a.optimizer, device=a.device, noise_from=a.noise_from,
                  x0=x0, seed=a.seed, stages=stages, lr=lr,
                  ckpt={'prefix': f"{ckdir}/{m}", 'every': 25},
                  mitigate=None if a.mitigate is None else
                  [s.strip() for s in a.mitigate.split(',')],
                  resume=a.resume,
                  parallel='auto' if a.parallel == 'auto' else int(a.parallel))
        times[m] = time.perf_counter() - t0
        results[m] = r
        # PER-METHOD immediate save: a kill after this leaves this method fully on disk
        np.savetxt(f"{a.out}_params_{m}.csv", r['x_opt'], delimiter=',')
        np.savetxt(f"{a.out}_density_{m}.csv", r['rho'], delimiter=',')
        np.savetxt(f"{a.out}_history_{m}.csv", np.array(r['history']),
                   delimiter=',', header='cost,L2,trace', comments='')
        print(f"  {m:9s}: {r['evals']:4d} iters | E={r['energy']:+.5f} "
              f"(lam_min={problem.lam_min:+.5f}) | L2={r['L2']:.4f} | trace={r['trace']:.4f}"
              f" | {times[m]:7.1f}s  [saved]", flush=True)
    # convergence figure + density figure + CSVs
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        matplotlib.rcParams['pdf.fonttype'] = 42
        col = {'classical': 'k', 'SD': 'r', 'FD': 'g', 'PO': 'b'}
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for m, r in results.items():
            arr = np.array(r['history'])
            for k in range(3):
                ax[k].plot(arr[:, k], col.get(m, 'C0'), lw=1.3, label=m)
        ax[0].axhline(problem.lam_min, color='gray', ls=':', lw=1.2, label=r'$\lambda_{min}$')
        ax[0].set_ylabel('cost'); ax[0].set_title('(a) cost')
        ax[1].set_ylabel('$L_2$'); ax[1].set_yscale('log'); ax[1].set_title('(b) $L_2$ error')
        ax[2].set_ylabel('trace distance'); ax[2].set_yscale('log'); ax[2].set_title('(c) trace distance')
        for k in range(3):
            ax[k].set_xlabel('iteration'); ax[k].grid(alpha=.3); ax[k].legend()
        fig.suptitle(problem.label)
        plt.tight_layout(); plt.savefig(f"{a.out}_convergence.png", dpi=130)
        print(f"[QConfLib] saved {a.out}_convergence.png")

        plt.figure(figsize=(7, 4))
        plt.plot(np.abs(problem.ground)**2, 'k-', lw=1.8, label='exact ground density')
        for m, r in results.items():
            if m != 'classical':
                plt.plot(r['rho'], 'o', ms=4, color=col.get(m, 'C0'),
                         label=f"shot readout ({m})")
        plt.xlabel('grid index'); plt.ylabel(r'$\rho_i=|\psi_i|^2$')
        plt.grid(alpha=.3); plt.legend(); plt.title(f"density — {problem.label}")
        plt.tight_layout(); plt.savefig(f"{a.out}_density.png", dpi=130)
        print(f"[QConfLib] saved {a.out}_density.png")
    except Exception as e:
        print(f"[QConfLib] (plots skipped: {e})")

    # (per-method CSVs already written immediately after each solve above)
    import json
    meta = {'args': vars(a), 'label': problem.label, 'lam_min': problem.lam_min,
            'init': info,
            'terminal': {m: {'energy': r['energy'], 'L2': r['L2'], 'trace': r['trace'],
                             'evals': r['evals']} for m, r in results.items()},
            'time_s': times,
            'best_trace': {m: float(min(h[2] for h in r['history'])) for m, r in results.items()}}
    with open(f"{a.out}_meta.json", 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[QConfLib] saved params/density/history CSVs + meta.json with prefix {a.out}_")
    print("DONE")


if __name__ == "__main__":
    main()
