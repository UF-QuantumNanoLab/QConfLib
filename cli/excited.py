#!/usr/bin/env python
"""First excited state by variational quantum deflation.

Deflates against a ground state that must already exist: pass its parameter CSV with
--ground (any `solve.py --out <prefix>` run writes `<prefix>_params_<METHOD>.csv`). The
overlap penalty is measured by a compute-uncompute circuit, so the run costs one extra
circuit per evaluation and stays hardware-executable.

  python cli/excited.py --dim 1 --n 5 --u0 1.0 --depth 5 --method SD \
      --ground outputs/run_params_SD.csv --stages 200:262144 --out outputs/exc1d

  python cli/excited.py --dim 3 --nx 3 --ny 3 --nz 3 --depth 8 --method SD \
      --ground paper/data/prod/r_3d_8x8x8_params_SD.csv --keep 6 --screen 48 \
      --stages 200:65536,100:262144 --out outputs/exc3d
"""
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from qconflib import problem_1d, problem_2d, problem_3d, solve_excited  # noqa: E402
from qconflib.gradients import DEFAULT_STAGES  # noqa: E402


def parse_stages(s):
    if not s:
        return DEFAULT_STAGES
    return tuple((int(a), int(b)) for a, b in (p.split(':') for p in s.split(',')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=1, choices=[1, 2, 3])
    ap.add_argument('--n', type=int, default=5, help="[1D] number of qubits")
    ap.add_argument('--u0', type=float, default=0.0, help="[1D] potential strength in eV")
    ap.add_argument('--nx', type=int, default=3)
    ap.add_argument('--ny', type=int, default=3)
    ap.add_argument('--nz', type=int, default=3)
    ap.add_argument('--t1', type=float, default=1 / 0.19)
    ap.add_argument('--t2', type=float, default=1 / 0.98)
    ap.add_argument('--t3', type=float, default=1.0)
    ap.add_argument('--method', default='SD', choices=['classical', 'SD', 'FD', 'PO'])
    ap.add_argument('--depth', type=int, default=5)
    ap.add_argument('--ground', required=True, help="CSV of the trained ground-state params")
    ap.add_argument('--beta', type=float, default=None, help="deflation strength")
    ap.add_argument('--shots', type=int, default=2**17, help="final density readout")
    ap.add_argument('--stages', default=None, help="steps:shots[,steps:shots...]")
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--seed', type=int, default=None, help="frozen simulator seed (CRN)")
    ap.add_argument('--device', default='auto', choices=['auto', 'GPU', 'CPU'])
    ap.add_argument('--noise-from', default=None, help="IBM calibration snapshot")
    ap.add_argument('--mitigate', default=None, help="readout[,zne]")
    ap.add_argument('--screen', type=int, default=16, help="raw inits screened classically")
    ap.add_argument('--keep', type=int, default=1,
                    help=">1 enables screen-and-refine (needed for narrow 2-D/3-D basins)")
    ap.add_argument('--screen-ref', action='store_true',
                    help="score seeds by trace to the exact 1st excited state (1-D)")
    ap.add_argument('--out', default='outputs/excited')
    a = ap.parse_args()

    if a.dim == 1:
        prob = problem_1d(a.n, a.u0)
    elif a.dim == 2:
        prob = problem_2d(a.nx, a.ny, a.t1, a.t2)
    else:
        prob = problem_3d(a.nx, a.ny, a.nz, a.t1, a.t2, a.t3)

    theta0 = np.loadtxt(a.ground, delimiter=',')
    kw = {}
    if a.lr is not None:
        kw['lr'] = a.lr
    t0 = time.perf_counter()
    r = solve_excited(prob, theta0, method=a.method, depth=a.depth, beta=a.beta,
                      shots=a.shots, device=a.device, noise_from=a.noise_from,
                      seed=a.seed, stages=parse_stages(a.stages), screen=a.screen,
                      keep=a.keep, screen_ref=prob.excited1 if a.screen_ref else None,
                      mitigate=None if not a.mitigate else a.mitigate.split(','), **kw)
    dt = time.perf_counter() - t0

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    np.savetxt(f'{a.out}_params.csv', r['x_opt'], delimiter=',')
    np.savetxt(f'{a.out}_density.csv', r['rho'], delimiter=',', header='rho', comments='')
    np.savetxt(f'{a.out}_history.csv', np.array(r['history']), delimiter=',',
               header='subspace_trace', comments='')
    meta = {'args': vars(a), 'label': prob.label, 'beta': r['beta'], 'init': r['init'],
            'terminal': {'energy': r['energy'], 'E1_exact': r['E1_exact'],
                         'fidelity': r['fidelity'], 'trace': r['trace'],
                         'degeneracy': r['degeneracy'], 'overlap0': r['overlap0']},
            'time_s': dt}
    json.dump(meta, open(f'{a.out}_meta.json', 'w'), indent=1)
    print(f"{a.method}: E1={r['energy']:.5f} (exact {r['E1_exact']:.5f}, degeneracy "
          f"{r['degeneracy']}) | subspace fidelity {r['fidelity']:.4f} | trace "
          f"{r['trace']:.4f} | <psi0|psi1>^2 {r['overlap0']:.1e} | beta {r['beta']:.3f} "
          f"| {dt:.0f}s")
    print(f"[QConfLib] saved params/density/history CSVs + meta.json with prefix {a.out}_")
    print('DONE')


if __name__ == '__main__':
    main()
