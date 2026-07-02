"""M6 (optional): cross-check BraKet's simulator against Qiskit.

Our other tests verify BraKet against *analytical* results (exact Bell/GHZ
amplitudes). This file adds an independent check against a mature third-party
simulator, Qiskit, for a spread of circuits — including the fully native,
decomposed form, which exercises the X90/RZ rewrites end to end.

The whole file is skipped automatically if qiskit isn't installed, so the core
suite never depends on it.

One wrinkle: bit ordering. BraKet treats q0 as the MOST significant bit, while
Qiskit treats q0 as the LEAST significant. So a Qiskit bitstring is the reverse
of a BraKet one, and we reverse Qiskit's keys before comparing.
"""

import math

import pytest

from braket.ast_nodes import Gate
from braket.decomposer import decompose
from braket.optimizer import optimize
from braket.parser import parse
from braket.simulator import StateVector

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def braket_prob_dict(circuit, tol=1e-12):
    """BraKet probabilities as {bitstring (q0 first): prob}, tiny entries dropped."""
    sv = StateVector(circuit.n_qubits)
    sv.run(circuit)
    probs = sv.probabilities()
    n = circuit.n_qubits
    return {
        format(i, f"0{n}b"): float(p)
        for i, p in enumerate(probs)
        if p > tol
    }


def to_qiskit(circuit):
    """Translate a BraKet `Circuit`'s gates into an equivalent Qiskit circuit."""
    qc = QuantumCircuit(circuit.n_qubits)
    for g in circuit.gates:
        q = g.qubits
        if g.name == "H":
            qc.h(q[0])
        elif g.name == "X":
            qc.x(q[0])
        elif g.name == "Y":
            qc.y(q[0])
        elif g.name == "Z":
            qc.z(q[0])
        elif g.name == "S":
            qc.s(q[0])
        elif g.name == "SDG":
            qc.sdg(q[0])
        elif g.name == "T":
            qc.t(q[0])
        elif g.name == "TDG":
            qc.tdg(q[0])
        elif g.name in ("SX", "X90"):
            qc.sx(q[0])  # SX == X90 == Rx(pi/2)
        elif g.name == "SXDG":
            qc.sxdg(q[0])
        elif g.name == "ID":
            qc.id(q[0])
        elif g.name == "RX":
            qc.rx(g.params[0], q[0])
        elif g.name == "RY":
            qc.ry(g.params[0], q[0])
        elif g.name == "RZ":
            qc.rz(g.params[0], q[0])
        elif g.name == "P":
            qc.p(g.params[0], q[0])
        elif g.name == "CNOT":
            qc.cx(q[0], q[1])
        elif g.name == "CZ":
            qc.cz(q[0], q[1])
        elif g.name == "CY":
            qc.cy(q[0], q[1])
        elif g.name == "SWAP":
            qc.swap(q[0], q[1])
        else:
            raise AssertionError(f"no Qiskit mapping for gate {g.name!r}")
    return qc


def qiskit_prob_dict(circuit, tol=1e-12):
    """Qiskit probabilities, re-keyed to BraKet's q0-first bit order."""
    sv = Statevector.from_instruction(to_qiskit(circuit))
    raw = sv.probabilities_dict()
    # Qiskit keys are little-endian (q0 last); reverse to match BraKet (q0 first).
    return {key[::-1]: float(v) for key, v in raw.items() if v > tol}


def assert_prob_dicts_close(a, b, atol=1e-9):
    for key in set(a) | set(b):
        assert abs(a.get(key, 0.0) - b.get(key, 0.0)) < atol, (
            f"probability mismatch at {key!r}: braket={a.get(key, 0.0)} "
            f"qiskit={b.get(key, 0.0)}"
        )


# --------------------------------------------------------------------------- #
# Cross-check circuits
# --------------------------------------------------------------------------- #

CIRCUITS = [
    "qubits 1\nH q0\n",
    "qubits 1\nX q0\nY q0\nZ q0\nRZ(0.9) q0\n",
    "qubits 2\nH q0\nCNOT q0 q1\n",                     # Bell
    "qubits 3\nH q0\nCNOT q0 q1\nCNOT q1 q2\n",         # GHZ
    "qubits 3\nH q0\nH q1\nH q2\nRZ(0.5) q1\nCNOT q0 q2\n",
    "qubits 2\nX q0\nRZ(1.3) q0\nH q1\nCNOT q1 q0\n",
    # Extended gate set:
    "qubits 1\nH q0\nS q0\nT q0\nSX q0\nP(0.8) q0\n",
    "qubits 1\nRX(0.7) q0\nRY(1.1) q0\nRZ(0.4) q0\n",
    "qubits 2\nH q0\nCZ q0 q1\nH q1\n",
    "qubits 2\nH q0\nCY q0 q1\n",
    "qubits 3\nH q0\nH q1\nSWAP q0 q2\nCZ q1 q2\nRY(0.9) q0\n",
]


@pytest.mark.parametrize("src", CIRCUITS)
def test_parsed_circuit_matches_qiskit(src):
    circuit = parse(src)
    assert_prob_dicts_close(braket_prob_dict(circuit), qiskit_prob_dict(circuit))


@pytest.mark.parametrize("src", CIRCUITS)
def test_optimized_circuit_matches_qiskit(src):
    circuit = optimize(parse(src))
    assert_prob_dicts_close(braket_prob_dict(circuit), qiskit_prob_dict(circuit))


@pytest.mark.parametrize("src", CIRCUITS)
def test_native_decomposition_matches_qiskit(src):
    # decompose() doesn't permute qubits (no routing), so the native circuit is
    # directly comparable. This validates the X90/RZ rewrites against Qiskit.
    circuit = decompose(parse(src))
    assert_prob_dicts_close(braket_prob_dict(circuit), qiskit_prob_dict(circuit))


def test_crosscheck_would_catch_a_wrong_result():
    # Sanity on the harness itself: a genuinely different circuit must NOT match,
    # so a passing cross-check above is meaningful.
    bell = braket_prob_dict(parse("qubits 2\nH q0\nCNOT q0 q1\n"))
    separable = qiskit_prob_dict(parse("qubits 2\nH q0\nH q1\n"))
    with pytest.raises(AssertionError):
        assert_prob_dicts_close(bell, separable)
