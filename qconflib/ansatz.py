import numpy as np
from qiskit import QuantumCircuit


def make_HEA(nq, depth, params):
    qc = QuantumCircuit(nq)
    for d in range(depth):
        for i in range(nq):
            qc.ry(params[i + d * nq], i)
        for i in range(nq - 1):
            qc.cx(i, i + 1)
    return qc


def QFT_LNN(qc, num_qubits, qubit_index):
    for idx in range(1, num_qubits):
        qc.rz(np.pi * ((2**idx - 1) / 2**(idx + 1)),
              qc.qubits[num_qubits - 1 - idx + qubit_index])
    for block_number in range(num_qubits - 1):
        qc.h(num_qubits - 1 - block_number + qubit_index)
        if 1 <= block_number <= num_qubits - 2:
            qc.cx(num_qubits - 1 - block_number + qubit_index,
                  num_qubits - 2 - block_number + qubit_index)
        else:
            for idx in range(num_qubits - block_number - 1):
                qc.cx(num_qubits - 1 - (num_qubits - 2 - idx) + qubit_index,
                      num_qubits - 1 - (num_qubits - idx - 1) + qubit_index)
        for idx in range(num_qubits - block_number - 2):
            qc.cx(num_qubits - 1 - (idx + 1 + block_number) + qubit_index,
                  num_qubits - 1 - (idx + 2 + block_number) + qubit_index)
        for idx in range(1, num_qubits - block_number):
            qc.rz(-np.pi / (2**(idx + 1)),
                  qc.qubits[num_qubits - 1 - (idx + block_number) + qubit_index])
        for idx in range(num_qubits - block_number - 1):
            qc.cx(num_qubits - 1 - (num_qubits - 2 - idx) + qubit_index,
                  num_qubits - 1 - (num_qubits - idx - 1) + qubit_index)
        if block_number <= num_qubits - 3:
            qc.cx(num_qubits - 1 - (1 + block_number) + qubit_index,
                  num_qubits - 1 - (2 + block_number) + qubit_index)
        else:
            for idx in range(num_qubits - block_number - 2):
                qc.cx(num_qubits - 1 - (idx + 1 + block_number) + qubit_index,
                      num_qubits - 1 - (idx + 2 + block_number) + qubit_index)
    for idx in range(1, num_qubits):
        qc.rz(np.pi * ((2**idx - 1) / 2**(idx + 1)),
              qc.qubits[num_qubits - 1 - (num_qubits - idx - 1) + qubit_index])
    qc.h(qubit_index)
    return qc


def increment(qc, m, start=0):
    for k in range(m - 1, 0, -1):
        qc.mcx([start + j for j in range(k)], start + k)
    qc.x(start)
    return qc
