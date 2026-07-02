"""M2 simulator tests.

These check the simulator against *known analytical results* — exact amplitudes
and probabilities — rather than merely "it ran". That's how we gain confidence
the quantum math is right before building the compiler stages on top of it.
"""

import math

import numpy as np
import pytest

from braket.parser import parse
from braket.simulator import StateVector, simulate

INV_SQRT2 = 1 / math.sqrt(2)


# --------------------------------------------------------------------------- #
# Single-qubit sanity checks
# --------------------------------------------------------------------------- #

def test_initial_state_is_all_zero():
    sv = StateVector(3)
    assert sv.state[0] == 1.0
    assert np.count_nonzero(sv.state) == 1


def test_x_flips_zero_to_one():
    sv = StateVector(1)
    sv.apply("X", (0,))
    assert np.allclose(sv.state, [0, 1])
    assert np.allclose(sv.probabilities(), [0, 1])


def test_hadamard_makes_equal_superposition():
    sv = StateVector(1)
    sv.apply("H", (0,))
    assert np.allclose(sv.state, [INV_SQRT2, INV_SQRT2])
    assert np.allclose(sv.probabilities(), [0.5, 0.5])


def test_z_is_a_phase_flip_only():
    # Z on |1> gives -|1>: the probability is unchanged, only the phase flips.
    sv = StateVector(1)
    sv.apply("X", (0,))   # -> |1>
    sv.apply("Z", (0,))   # -> -|1>
    assert np.allclose(sv.state, [0, -1])
    assert np.allclose(sv.probabilities(), [0, 1])


def test_y_on_zero():
    # Y|0> = i|1>.
    sv = StateVector(1)
    sv.apply("Y", (0,))
    assert np.allclose(sv.state, [0, 1j])
    assert np.allclose(sv.probabilities(), [0, 1])


def test_rz_applies_expected_relative_phase():
    # RZ(pi) = diag(-i, i). On |1> that gives i|1>.
    sv = StateVector(1)
    sv.apply("X", (0,))            # -> |1>
    sv.apply("RZ", (0,), (math.pi,))
    assert np.allclose(sv.state, [0, 1j])


def test_h_twice_is_identity():
    sv = StateVector(1)
    sv.apply("H", (0,)).apply("H", (0,))
    assert np.allclose(sv.state, [1, 0])


# --------------------------------------------------------------------------- #
# Entanglement: Bell and GHZ
# --------------------------------------------------------------------------- #

def test_bell_state_amplitudes_and_marginal():
    sv = StateVector(2)
    sv.apply("H", (0,)).apply("CNOT", (0, 1))
    # (|00> + |11>) / sqrt(2): amplitudes at indices 0 and 3.
    assert np.allclose(sv.state, [INV_SQRT2, 0, 0, INV_SQRT2])

    marginal = sv.marginal([0, 1])
    assert set(marginal) == {"00", "11"}
    assert math.isclose(marginal["00"], 0.5)
    assert math.isclose(marginal["11"], 0.5)
    # The forbidden anti-correlated outcomes never occur.
    assert "01" not in marginal and "10" not in marginal


def test_ghz_three_qubit_state():
    sv = StateVector(3)
    sv.apply("H", (0,)).apply("CNOT", (0, 1)).apply("CNOT", (1, 2))
    expected = np.zeros(8, dtype=complex)
    expected[0] = INV_SQRT2   # |000>
    expected[7] = INV_SQRT2   # |111>
    assert np.allclose(sv.state, expected)


# --------------------------------------------------------------------------- #
# Generality: any qubit pair, adjacent or not
# --------------------------------------------------------------------------- #

def test_cnot_on_nonadjacent_qubits():
    # X on q0, then CNOT q0->q2 should flip q2, giving basis |101> (index 5).
    sv = StateVector(3)
    sv.apply("X", (0,)).apply("CNOT", (0, 2))
    assert np.argmax(np.abs(sv.state)) == 0b101
    assert np.isclose(sv.state[0b101], 1.0)


def test_cnot_with_control_greater_than_target():
    # Control q2, target q0: X on q2 then CNOT q2->q0 -> q0 and q2 set: |101>.
    sv = StateVector(3)
    sv.apply("X", (2,)).apply("CNOT", (2, 0))
    assert np.isclose(sv.state[0b101], 1.0)


def test_swap_exchanges_two_qubits():
    # X on q0 -> |10>; SWAP q0,q1 -> |01> (index 1).
    sv = StateVector(2)
    sv.apply("X", (0,)).apply("SWAP", (0, 1))
    assert np.isclose(sv.state[0b01], 1.0)


def test_single_gate_on_high_index_qubit():
    # H on the last qubit of a 3-qubit register: superposition on q2 only.
    sv = StateVector(3)
    sv.apply("H", (2,))
    assert np.allclose(sv.state, [INV_SQRT2, INV_SQRT2, 0, 0, 0, 0, 0, 0])


def test_state_stays_normalized():
    sv = StateVector(3)
    for gate, qs in [("H", (0,)), ("CNOT", (0, 1)), ("RZ", (2,)), ("Y", (2,))]:
        params = (0.7,) if gate == "RZ" else ()
        sv.apply(gate, qs, params)
    assert math.isclose(float(np.sum(sv.probabilities())), 1.0, rel_tol=1e-12)


# --------------------------------------------------------------------------- #
# Sampling and the end-to-end simulate() helper
# --------------------------------------------------------------------------- #

def test_sampling_is_deterministic_with_seed():
    sv = StateVector(2)
    sv.apply("H", (0,)).apply("CNOT", (0, 1))
    counts_a = sv.sample([0, 1], shots=500, seed=42)
    counts_b = sv.sample([0, 1], shots=500, seed=42)
    assert counts_a == counts_b


def test_bell_sampling_only_correlated_outcomes():
    sv = StateVector(2)
    sv.apply("H", (0,)).apply("CNOT", (0, 1))
    counts = sv.sample([0, 1], shots=1000, seed=7)
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 1000
    # Both correlated outcomes should show up with a fair coin (~500 each).
    assert counts.get("00", 0) > 300 and counts.get("11", 0) > 300


def test_simulate_parses_and_measures_declared_qubits():
    circuit = parse("qubits 2\nH q0\nCNOT q0 q1\nMEASURE q0\nMEASURE q1\n")
    result = simulate(circuit, shots=200, seed=1)
    assert result.measured_qubits == [0, 1]
    assert set(result.counts) <= {"00", "11"}
    assert sum(result.counts.values()) == 200
    assert np.allclose(result.probabilities, [0.5, 0, 0, 0.5])


def test_simulate_without_measure_samples_all_qubits():
    circuit = parse("qubits 2\nX q0\n")
    result = simulate(circuit, shots=10, seed=1)
    assert result.measured_qubits == [0, 1]
    # X q0 -> deterministic |10>.
    assert result.counts == {"10": 10}
