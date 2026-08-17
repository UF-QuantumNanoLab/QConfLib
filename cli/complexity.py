#!/usr/bin/env python
"""QConfLib CLI — gate / measurement complexity scan (no simulation, pure transpilation).

For each method, total gates for ONE <M_box> evaluation (1D, all its circuits):
logical = transpiled to {u,cx}; native = IBM native basis + LNN line coupling (opt level 1).

Examples:
  python cli/complexity.py                   # n = 2..12, depth 2
  python cli/complexity.py --nmax 14 --depth 3
"""
import argparse
import os
import sys
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qconflib import make_HEA, QFT_LNN, increment, NATIVE_BASIS  # noqa: E402


def sd_circuits(ansatz, n):
    out = []
    for j in range(n):
        m = n - j
        qc = QuantumCircuit(n, m)
        qc.append(ansatz.to_gate(label='U'), qc.qubits)
        for i in range(n - j - 1):
            qc.cx(i + 1, i)
        qc.h(n - 1 - j); qc.measure(range(m), range(m))
        out.append(qc)
    return out


def fd_circuits(ansatz, n):
    qc = QuantumCircuit(n + 1, 1)
    qc.append(ansatz.to_gate(label='U'), qc.qubits[1:])
    QFT_LNN(qc, n, 1); qc.h(0)
    for idx in range(n):
        qc.cp((2 * np.pi / (2**n)) * (2**idx), 0, n - idx)
    qc.h(0); qc.measure(0, 0)
    qc2 = QuantumCircuit(n, n)
    qc2.append(ansatz.to_gate(label='U'), qc2.qubits)
    increment(qc2, n); qc2.h(0); qc2.measure(range(n), range(n))
    return [qc, qc2]


def po_circuits(ansatz, n):
    c1 = QuantumCircuit(n, 1)
    c1.append(ansatz.to_gate(label='U'), c1.qubits)
    c1.h(0); c1.measure(0, 0)
    c2 = QuantumCircuit(n, n)
    c2.append(ansatz.to_gate(label='U'), c2.qubits)
    increment(c2, n); c2.h(0); c2.measure(range(n), range(n))
    return [c1, c2]


CIRCUITS = {'PO': po_circuits, 'SD': sd_circuits, 'FD': fd_circuits}
N_MEAS = {'PO': lambda n: 2, 'SD': lambda n: n, 'FD': lambda n: 2}


def count_gates(tqc):
    return sum(v for k, v in dict(tqc.count_ops()).items() if k not in ('measure', 'barrier'))


# Sabre routing is randomised: without a fixed seed the PO/FD counts move by up to ~1% run to
# run, since equal-cost SWAP placements leave different 1q gates mergeable. SD needs no
# routing on a line and is stable either way.
SEED_TRANSPILER = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nmin', type=int, default=2)
    ap.add_argument('--nmax', type=int, default=12)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--out', default='outputs/complexity')
    a = ap.parse_args()
    ns = list(range(a.nmin, a.nmax + 1))
    methods = ['PO', 'SD', 'FD']
    data = {m: {'logical': [], 'native': [], 'percirc': [], 'percirc_max': []}
            for m in methods}
    print(f"[complexity] depth={a.depth}   native gates: sum over one evaluation / per circuit")
    print(f"{'n':>3} | " + " | ".join(f"{m + ' sum/circ':>20}" for m in methods))
    for n in ns:
        ansatz = make_HEA(n, a.depth, np.full(n * a.depth, 0.7))
        row = []
        for m in methods:
            lg, per = 0, []
            for qc in CIRCUITS[m](ansatz, n):
                lg += count_gates(transpile(qc, basis_gates=['u', 'cx'], optimization_level=1,
                                            seed_transpiler=SEED_TRANSPILER))
                per.append(count_gates(transpile(qc, basis_gates=NATIVE_BASIS,
                                                 coupling_map=CouplingMap.from_line(qc.num_qubits),
                                                 optimization_level=1,
                                                 seed_transpiler=SEED_TRANSPILER)))
            data[m]['logical'].append(lg); data[m]['native'].append(sum(per))
            data[m]['percirc'].append(np.mean(per)); data[m]['percirc_max'].append(max(per))
            row.append(f"{sum(per):>7}/{np.mean(per):>6.0f}")
        print(f"{n:>3} | " + " | ".join(f"{r:>20}" for r in row), flush=True)

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(f"{a.out}.csv", 'w') as f:
        f.write("n,N," + ",".join(f"{m}_logical,{m}_native,{m}_percirc,{m}_percirc_max"
                                  for m in methods)
                + "," + ",".join(f"{m}_nmeas" for m in methods) + "\n")
        for i, n in enumerate(ns):
            f.write(f"{n},{2**n},"
                    + ",".join(f"{data[m]['logical'][i]},{data[m]['native'][i]},"
                               f"{data[m]['percirc'][i]:.1f},{data[m]['percirc_max'][i]}"
                               for m in methods)
                    + "," + ",".join(str(N_MEAS[m](n)) for m in methods) + "\n")

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    matplotlib.rcParams['pdf.fonttype'] = 42
    col = {'PO': 'tab:blue', 'SD': 'tab:red', 'FD': 'tab:purple'}
    mk = {'PO': 'o', 'SD': 's', 'FD': '^'}
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    for m in methods:
        ax[0].plot(ns, data[m]['logical'], '-', color=col[m], lw=2, label=m)
        ax[1].plot(ns, np.log2(data[m]['logical']), '-', color=col[m], lw=2)
        ax[1].plot(ns, np.log2(data[m]['native']), mk[m], color=col[m], ms=6,
                   label=f'{m} (native)')
        ax[2].plot(ns, [N_MEAS[m](n) for n in ns], '-', marker=mk[m], color=col[m], lw=2,
                   ms=5, label=f"{m}: " + (r'$O(n)$' if m == 'SD' else r'$O(1)$'))
    ax[1].plot([], [], 'k-', lw=2, label='logical (lines)')
    ax[0].set_xlabel(r'$\log_2 N$'); ax[0].set_ylabel(r'$C_{total}$ (logical)')
    ax[0].set_title(r'(a) logical gates per $\langle M_{box}\rangle$')
    ax[1].set_xlabel(r'$\log_2 N$'); ax[1].set_ylabel(r'$\log_2 C_{total}$')
    ax[1].set_title('(b) native (LNN) vs logical')
    ax[2].set_xlabel(r'$\log_2 N$'); ax[2].set_ylabel('circuits per evaluation')
    ax[2].set_title('(c) measurement count')
    for k in range(3):
        ax[k].grid(alpha=.3); ax[k].legend()
    fig.suptitle(f'QConfLib complexity scan (1D hard-wall, depth {a.depth}, LNN)')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches='tight')
    print(f"[complexity] saved {a.out}.csv / .png / .pdf")
    print("DONE")


if __name__ == "__main__":
    main()
