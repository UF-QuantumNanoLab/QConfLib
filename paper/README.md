# Archived results

The runs used in the paper. The notebooks in the repository root read from here, so the production-scale results can be inspected without repeating hours of simulation or spending QPU time.

```
data/       training runs, one group of files per run:
              <tag>_history_<METHOD>.csv    (cost, L2, trace) per iteration
              <tag>_density_<METHOD>.csv    shot-based density at the optimum
              <tag>_params_<METHOD>.csv     trained parameters
              <tag>_meta.json               the full argument namespace of the run, plus
                                            seeds, timings, terminal metrics and lam_min
            complexity_scan.csv             gate and measurement counts vs qubit count
hw_run/     the IBM hardware campaign, one group per block:
              <block>.json                  job id, backend, shots, physical qubits
              <block>_counts_raw.json       raw measurement counts, exactly as returned
              <block>_circuits.qpy          the transpiled circuits that produced them
              <block>_results.json          decoded results
            manifest.json                   pre-flight circuit budget for both candidates
```

Because `meta.json` stores the argument namespace, no command line has to be remembered:

```python
import json
json.load(open('paper/data/prod/r_2d_32x32_staged_meta.json'))['args']
```

## The hardware blocks

One ten-minute public allocation on `ibm_kingston` (Heron r2, 156 qubits), submitted as eight
jobs. Every hardware number in the paper is recomputable from these counts.

| block | content |
|---|---|
| B | 1-D N=32 core matrix at the noiseless optimum: SD/PO/FD estimators plus density, with in-job readout calibration |
| C | block B repeated in a different calibration slot, to measure drift |
| D, E, F | 2-D 32x32 and 3-D 8x8x8 ground states, and first excited states in all three dimensions |
| G | block B with dynamical decoupling and Pauli twirling |
| H | 12 randomly initialized parameter sets, the protocol of the Poisson benchmark |
| I | density readout versus shot count, 1-D ground and first excited |
| J | parameter provenance: the same state prepared at parameters from four training provenances |

Notebook `06_real_hardware.ipynb` walks through all of it, and runs offline.
