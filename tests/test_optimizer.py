"""M3 optimizer tests.

Two things to prove for every rule:
  1. It fires when it should (and *doesn't* when it shouldn't) — checked by
     inspecting the resulting gate list.
  2. It preserves meaning — checked by simulating before/after and comparing
     states (up to global phase) and probabilities (exactly).
"""

import math

import pytest

from braket.ast_nodes import Circuit, Gate, Measure
from braket.optimizer import optimize
from braket.parser import parse
from tests.helpers import (
    gate_summary,
    probabilities_of,
    state_of,
    states_equal_up_to_global_phase,
)


def optimize_source(src):
    return optimize(parse(src))


# --------------------------------------------------------------------------- #
# Rule (a): self-inverse cancellation
# --------------------------------------------------------------------------- #

def test_hadamard_pair_cancels():
    out = optimize_source("qubits 1\nH q0\nH q0\n")
    assert out.gates == []


@pytest.mark.parametrize("gate", ["X", "Y", "Z", "H"])
def test_single_qubit_self_inverse_pairs_cancel(gate):
    out = optimize_source(f"qubits 1\n{gate} q0\n{gate} q0\n")
    assert out.gates == []


def test_cnot_pair_cancels():
    out = optimize_source("qubits 2\nCNOT q0 q1\nCNOT q0 q1\n")
    assert out.gates == []


def test_reversed_cnot_does_not_cancel():
    # CNOT q0 q1 then CNOT q1 q0 are different operations; nothing cancels.
    out = optimize_source("qubits 2\nCNOT q0 q1\nCNOT q1 q0\n")
    assert gate_summary(out) == [("CNOT", (0, 1), ()), ("CNOT", (1, 0), ())]


def test_cancellation_sees_through_disjoint_gate():
    # X on q1 sits between the two H q0; it commutes past, so the H's still cancel.
    out = optimize_source("qubits 2\nH q0\nX q1\nH q0\n")
    assert gate_summary(out) == [("X", (1,), ())]


def test_no_cancellation_when_same_qubit_gate_between():
    # Z q0 between the two H q0 blocks cancellation.
    out = optimize_source("qubits 1\nH q0\nZ q0\nH q0\n")
    assert gate_summary(out) == [("H", (0,), ()), ("Z", (0,), ()), ("H", (0,), ())]


def test_measurement_is_a_barrier():
    out = optimize_source("qubits 1\nH q0\nMEASURE q0\nH q0\n")
    # Both H's survive because the measurement sits between them.
    assert [g.name for g in out.gates] == ["H", "H"]
    assert any(isinstance(i, Measure) for i in out.instructions)


def test_odd_run_leaves_one_gate():
    out = optimize_source("qubits 1\nX q0\nX q0\nX q0\n")
    assert gate_summary(out) == [("X", (0,), ())]


def test_even_run_fully_cancels_via_fixpoint():
    out = optimize_source("qubits 1\nH q0\nH q0\nH q0\nH q0\n")
    assert out.gates == []


def test_nested_cancellation():
    # Inner X;X cancels, then the outer H;H become adjacent and cancel too.
    out = optimize_source("qubits 1\nH q0\nX q0\nX q0\nH q0\n")
    assert out.gates == []


# --------------------------------------------------------------------------- #
# Rule (b): RZ merging
# --------------------------------------------------------------------------- #

def test_adjacent_rz_merge_into_sum():
    out = optimize_source("qubits 1\nRZ(0.3) q0\nRZ(0.4) q0\n")
    summary = gate_summary(out)
    assert len(summary) == 1
    name, qubits, params = summary[0]
    assert name == "RZ" and qubits == (0,)
    assert math.isclose(params[0], 0.7)


def test_three_rz_merge_via_fixpoint():
    out = optimize_source("qubits 1\nRZ(0.1) q0\nRZ(0.2) q0\nRZ(0.3) q0\n")
    (only,) = out.gates
    assert only.name == "RZ"
    assert math.isclose(only.params[0], 0.6)


def test_rz_canceling_to_zero_is_dropped():
    out = optimize_source("qubits 1\nRZ(0.5) q0\nRZ(-0.5) q0\n")
    assert out.gates == []


def test_rz_summing_to_2pi_is_dropped_up_to_global_phase():
    src = "qubits 1\nX q0\nRZ(3.14159265358979) q0\nRZ(3.14159265358979) q0\n"
    original = parse(src)
    out = optimize(original)
    # The two RZ(pi) sum to 2*pi -> identity up to global phase -> dropped.
    assert [g.name for g in out.gates] == ["X"]
    assert states_equal_up_to_global_phase(state_of(original), state_of(out))


def test_rz_merge_does_not_cross_other_gate():
    # An H between the RZs blocks the merge.
    out = optimize_source("qubits 1\nRZ(0.3) q0\nH q0\nRZ(0.4) q0\n")
    assert [g.name for g in out.gates] == ["RZ", "H", "RZ"]


# --------------------------------------------------------------------------- #
# Semantics preservation on realistic circuits
# --------------------------------------------------------------------------- #

EQUIVALENCE_CIRCUITS = [
    # Bell state padded with redundant gates that should optimize away.
    "qubits 2\nH q0\nH q0\nH q0\nCNOT q0 q1\nCNOT q0 q1\nCNOT q0 q1\n",
    # RZ merges interleaved with a disjoint qubit.
    "qubits 2\nH q0\nRZ(0.4) q0\nX q1\nRZ(0.6) q0\nCNOT q0 q1\n",
    # GHZ with cancelling pairs sprinkled in.
    "qubits 3\nH q0\nX q2\nX q2\nCNOT q0 q1\nCNOT q1 q2\nZ q0\nZ q0\n",
    # Pure rotations that partially cancel.
    "qubits 1\nH q0\nRZ(1.1) q0\nRZ(-1.1) q0\nRZ(0.5) q0\n",
]


@pytest.mark.parametrize("src", EQUIVALENCE_CIRCUITS)
def test_optimize_preserves_state_and_probabilities(src):
    original = parse(src)
    optimized = optimize(original)

    # Meaning is unchanged...
    assert states_equal_up_to_global_phase(state_of(original), state_of(optimized))
    # ...and probabilities match exactly (phase-independent).
    assert probabilities_of(original) == pytest.approx(probabilities_of(optimized))


@pytest.mark.parametrize("src", EQUIVALENCE_CIRCUITS)
def test_optimize_never_grows_the_circuit(src):
    original = parse(src)
    optimized = optimize(original)
    assert len(optimized.gates) <= len(original.gates)


def test_measurements_are_preserved():
    src = "qubits 2\nH q0\nH q0\nCNOT q0 q1\nMEASURE q0\nMEASURE q1\n"
    out = optimize(parse(src))
    assert out.measured_qubits == [0, 1]
