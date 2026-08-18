# QConfLib

**QConfLib** is an open-source Python library for quantum-computing-based simulation of quantum confinement in semiconductor nanostructures. Built on **Qiskit**, it implements variational quantum algorithms for solving the discretized effective-mass Schrödinger equation in **one, two, and three dimensions**, including quantum wells (1D), nanowire cross sections (2D), and quantum dots (3D). Both ground and first excited states are supported, together with anisotropic effective masses and spatially varying electrostatic potentials.

The confinement Hamiltonian is discretized on a uniform finite-difference grid and encoded using a qubit register whose size grows logarithmically with the number of grid points. The trial wave function is represented by a hardware-efficient parameterized quantum circuit, and its energy is minimized through a quantum-classical variational optimization procedure. Excited states are obtained using variational quantum deflation (VQD).

QConfLib implements three approaches for measuring the kinetic-energy contribution to the confinement Hamiltonian:

* **PO (Permutation Operator):** uses a cyclic-shift/increment circuit to evaluate nearest-neighbor couplings.
* **SD (Sparse Decomposition):** decomposes the kinetic operator into `n` sparse components measured using shallow CNOT-ladder circuits.
* **FD (Fourier Diagonalization):** uses a quantum Fourier transform (QFT) to diagonalize the periodic kinetic operator, together with a Hadamard test and a correction for the Dirichlet boundary condition.

These methods provide different tradeoffs between **circuit depth and number of measurement circuits**, enabling systematic benchmarking under both sampling noise and realistic quantum-device noise.

A common execution interface allows the same confinement problem to be run on an **ideal statevector simulator**, a **calibrated device-noise emulator based on IBM hardware properties**, or **real IBM quantum processors**. QConfLib also supports hardware-aware optimization and in-loop noise-mitigation techniques, providing a reproducible platform for developing and benchmarking quantum algorithms for semiconductor quantum-confinement simulation.

This implementation builds on and extends the following works:

> **Ning Yang and Jing Guo** (2023). *A Quantum-Computing-Based Method for Solving Quantum Confinement Problem in Semiconductor*. IEEE Transactions on Electron Devices, 70(3), 1366–1373.

> **Yuki Sato, Ruho Kondo, Satoshi Koide, Hideki Takamatsu, and Nobuyuki Imoto** (2021). *Variational Quantum Algorithm Based on the Minimum Potential Energy for Solving the Poisson Equation*.

> **Hai-Ling Liu, Yu-Sen Wu, Lin-Chun Wan, Shi-Jie Pan, Su-Juan Qin, Fei Gao, and Qiao-Yan Wen** (2021). *Variational Quantum Algorithm for the Poisson Equation*.

> **Dongyun Chung, Jiyong Choi, and Jung-Il Choi** (2026). *VQA_POISSON: A Quantum Library for Solving Two-Dimensional Poisson Equations with Mixed Boundary Conditions*.

---

## Features

- Hybrid VQA framework for confinement eigenproblems in 1D / 2D / 3D
- Three measurement decompositions (PO / SD / FD), each validated against exact expectation values
- Anisotropic effective masses and optional sine-bump electrostatic potentials
- First excited states by deflation, with the overlap measured on hardware-executable circuits
- Parameter-shift gradients with Adam, and any `scipy.optimize.minimize` method for comparison
- In-loop readout correction and zero-noise extrapolation
- Ideal, IBM-calibration-noise, and live-hardware execution behind one interface

---

## Repository Structure

- `qconflib/` : source code for the quantum-classical hybrid solver
- `qconflib/operators.py` :
  Confinement Hamiltonians as Kronecker sums over directions. Provides `problem_1d/2d/3d` and the `Problem` class, which also holds the exact spectrum used as a reference.
- `qconflib/ansatz.py` :
  Qiskit-based circuit blocks: the RY + CNOT hardware-efficient ansatz, the LNN-optimized QFT of *Park & Ahn (2023)*, and the increment (permutation) operator of *Sato et al. (2021)*.
- `qconflib/estimators.py` :
  Sub-register estimators for the PO, SD, and FD decompositions, plus diagonal sampling for the density and the circuit-count helper `n_circuits`.
- `qconflib/gradients.py` :
  `BatchedExpectation`, which transpiles the measurement templates once and evaluates all `2P` parameter-shift circuits in a single batched run, and the Adam driver with its coarse-to-fine shot schedule.
- `qconflib/backends.py` :
  `Executor`, the single execution interface. Selects an ideal statevector simulator, a density-matrix simulator built from an IBM calibration snapshot, or a live device through `SamplerV2`, and owns the transpilation policy and the frozen simulator seed.
- `qconflib/mitigation.py` :
  Readout confusion-matrix inversion and self-inverse 2q gate folding with Richardson coefficients, both applied inside the training loop.
- `qconflib/vqe.py` : classical basin screening (`screen_init`) and the ground-state driver (`solve`).
- `qconflib/vqd.py` : first excited states by deflation (`solve_excited`), with the overlap measured by a compute-uncompute circuit.
- `qconflib/validate.py` : estimator validation against exact expectation values.

- `01_1d_box.ipynb` : quantum well. The Hamiltonian, the ansatz circuit, the three decompositions and the circuits they run, training under shot noise, COBYLA against parameter-shift Adam, and the effect of device noise.
- `02_2d_anisotropic.ipynb` : nanowire cross section with silicon anisotropic masses, and where a flat shot budget stops working.
- `03_3d_dot.ipynb` : quantum dot on nine qubits, with degenerate levels.
- `04_excited_states.ipynb` : first excited states by deflation.
- `05_error_mitigation.ipynb` : readout correction and zero-noise extrapolation inside the training loop.
- `06_real_hardware.ipynb` : running on an IBM device. Executes offline against the calibration twin and the archived counts, so no QPU allocation is needed.

- `cli/solve.py` : ground-state optimization from the command line. Writes convergence history, density, optimal parameters, and a `meta.json` recording the full argument namespace.
- `cli/excited.py` : the same for first excited states, deflating against a trained ground state.
- `cli/validate.py` : estimator validation. Exits non-zero on failure.
- `cli/complexity.py` : gate and measurement counts versus problem size, no simulation involved.

- `paper/data/` : archived training runs behind the paper figures, each with its full argument namespace in `meta.json`.
- `paper/hw_run/` : raw hardware counts, transpiled circuits, and job ids. The notebooks read from here.
- `env/` : optional CUDA-accelerated qiskit-aer (prebuilt wheel and build script).
- `outputs/` : where the CLI writes.

---

## Installation & Requirements

### Requirements

- Python >= 3.9
- qiskit >= 2.0
- qiskit-aer >= 0.17
- qiskit-ibm-runtime >= 0.40
- qiskit-algorithms >= 0.3
- NumPy
- SciPy
- Matplotlib

### Installation

```bash
pip install -r requirements.txt
```

**Optional GPU acceleration.** PyPI's `qiskit-aer-gpu` stops at 0.15.1 and supports qiskit 1.x only, so for qiskit 2.x the CUDA backend must be built from source. `env/` ships both forms:

```bash
pip install env/qiskit_aer-0.17.2-*.whl          # prebuilt: Python 3.11, sm_86, CUDA 12.x
CUDA_ARCH=8.9 bash env/build_aer_gpu.sh          # source build for another GPU arch
```

No flags are needed either way. The library runs with `device='auto'` and selects the GPU when the installed qiskit-aer supports it (`--device GPU/CPU` to force).

---

### Run Instructions

The six notebooks are the intended entry point and each runs end to end in a few minutes.

The same calculations are available from the command line. Run the estimator validation first, since a failure there invalidates everything downstream:

```bash
python cli/validate.py                      # PO/SD/FD against exact; exit 1 on failure
python cli/complexity.py --nmin 2 --nmax 12 --depth 2 --out outputs/complexity
python cli/solve.py --dim 1 --n 5 --depth 4 --method classical,SD,FD \
    --seed 111 --init-seed 29 --stages 300:1048576 --out outputs/r_1d_N32
```

The third command reproduces the paper's `N=32` run; `--help` lists all flags. Every archived run carries its own provenance, so any of them can be repeated without remembering a command line:

```python
import json
meta = json.load(open('paper/data/prod/r_1d_N32_meta.json'))
meta['args']        # the exact argument namespace cli/solve.py was invoked with
meta['terminal']    # the trace distance each estimator reached
```

The Python API in three lines:

```python
from qconflib import problem_1d, solve
r = solve(problem_1d(5, u0=1.0), method='SD', depth=4, seed=111)
print(r['energy'], r['trace']); rho = r['rho']
```

**Make sure your IBM Quantum API key is set properly before running the hardware notebook.** Save it once with `QiskitRuntimeService.save_account(...)`; it is never read from the repository.

---

## License

MIT License. See [`LICENSE`](LICENSE).

---

## Contact

For issues, suggestions, or collaborations, please contact: [**qimao.yang@ufl.edu**](mailto:qimao.yang@ufl.edu)
