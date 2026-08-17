"""Execution backends behind one interface: transpile(qc) and run_counts(circuits, shots).

  ideal      Executor()
  emulated   Executor(noise_from='sherbrooke')   density matrix + IBM calibration snapshot
  hardware   Executor(hardware='ibm_kingston')   live QPU through SamplerV2"""
import os
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

NATIVE_BASIS = ['ecr', 'id', 'rz', 'sx', 'x']


def resolve_device(device='auto'):
    """'auto' -> 'GPU' if this qiskit-aer build supports it, else 'CPU'."""
    if device != 'auto':
        return device
    try:
        if 'GPU' in AerSimulator(method='statevector').available_devices():
            return 'GPU'
    except Exception:
        pass
    return 'CPU'


def real_device_noise(name):
    """NoiseModel + native basis from an IBM calibration snapshot. Eagle is ECR, Heron CZ."""
    from qiskit_ibm_runtime import fake_provider as _fp
    cls = getattr(_fp, 'Fake' + ''.join(p.capitalize() for p in name.split('_')))
    be = cls()
    basis = [g for g in be.operation_names
             if g in ('cz', 'ecr', 'cx', 'rzz', 'rz', 'sx', 'x', 'id')]
    return NoiseModel.from_backend(be), basis, be.name


def fake_backend(name):
    """The calibration twin object, e.g. FakeKingston for 'ibm_kingston'."""
    from qiskit_ibm_runtime import fake_provider as _fp
    short = name[4:] if name.startswith('ibm_') else name
    return getattr(_fp, 'Fake' + ''.join(p.capitalize() for p in short.split('_')))()


class Executor:
    """Owns the simulator and the transpilation policy; estimators only call run_counts().

    seed freezes the simulator RNG (common random numbers). hardware names an IBM backend;
    local=True replays the same circuits on its calibration twin instead of the QPU."""

    def __init__(self, device='auto', noise_from=None, seed=None,
                 parallel_experiments='auto', hardware=None, local=False,
                 seed_transpiler=42, optimization_level=3):
        self.hardware = hardware
        if hardware:
            self._init_hardware(hardware, local, seed, seed_transpiler, optimization_level)
            return
        device = resolve_device(device)
        self.device = device
        self.noise_from = noise_from
        self.seed = seed
        kw = {} if seed is None else {'seed_simulator': int(seed)}
        if parallel_experiments == 'auto':
            parallel_experiments = min(24, os.cpu_count() or 1)
        self.parallel_experiments = parallel_experiments
        if parallel_experiments and parallel_experiments > 1:
            kw['max_parallel_experiments'] = int(parallel_experiments)
            kw['max_parallel_threads'] = int(os.cpu_count() or 1)
        if noise_from:
            nm, basis, name = real_device_noise(noise_from)
            self.sim = AerSimulator(method='density_matrix', device=device,
                                    noise_model=nm, **kw)
            self.basis = basis
            self.backend_name = name
            self.noisy = True
        else:
            self.sim = AerSimulator(method='statevector', device=device, **kw)
            self.basis = None
            self.backend_name = f'ideal statevector ({device})'
            self.noisy = False

    def _init_hardware(self, name, local, seed, seed_transpiler, optimization_level):
        self.device = 'QPU'
        self.noise_from = None
        self.noisy = True
        self.seed = seed
        self.local = local
        self.backend_name = name
        self.seed_transpiler = seed_transpiler
        self.optimization_level = optimization_level
        self.parallel_experiments = 1
        self.last_job_id = None
        if local:
            self.backend = fake_backend(name)
            self.sim = AerSimulator.from_backend(self.backend)
        else:
            from qiskit_ibm_runtime import QiskitRuntimeService
            self.backend = QiskitRuntimeService().backend(name)
            self.sim = None
        self.basis = list(self.backend.operation_names)

    def transpile(self, qc):
        """Call once on parameterised templates: re-transpiling every evaluation dominates
        the runtime below about 2^20 shots."""
        if self.hardware:
            return transpile(qc, backend=self.backend,
                             optimization_level=self.optimization_level,
                             seed_transpiler=self.seed_transpiler)
        if self.noisy:
            return transpile(qc, basis_gates=self.basis,
                             coupling_map=CouplingMap.from_line(qc.num_qubits),
                             optimization_level=1)
        return transpile(qc, self.sim, optimization_level=1)

    def counts(self, qc, shots):
        return self.run_counts([self.transpile(qc)], shots)[0]

    def run_counts(self, bound_circuits, shots):
        """One run for a list of transpiled, parameter-bound circuits; returns counts dicts."""
        if self.hardware:
            return self._run_hardware(bound_circuits, shots)
        counts = self.sim.run(bound_circuits, shots=shots).result().get_counts()
        return [counts] if isinstance(counts, dict) else counts

    def _run_hardware(self, circuits, shots):
        """Submit one job for the batch and block; submit/fetch below are the async form."""
        job = self.submit(circuits, shots)
        return self.fetch(job)

    def submit(self, circuits, shots):
        """Submit without waiting; the job id is kept on the Executor so it can be archived."""
        from qiskit_ibm_runtime import SamplerV2
        mode = self.backend if not self.local else self.backend
        job = SamplerV2(mode=mode).run(circuits, shots=shots)
        self.last_job_id = job.job_id()
        return job

    @staticmethod
    def fetch(job):
        """Counts from a SamplerV2 job, blocking until it finishes."""
        res = job.result()
        return [r.data[list(r.data.keys())[0]].get_counts() for r in res]

    def job(self, job_id):
        """Re-attach to a submitted job by id."""
        from qiskit_ibm_runtime import QiskitRuntimeService
        return QiskitRuntimeService().job(job_id)
