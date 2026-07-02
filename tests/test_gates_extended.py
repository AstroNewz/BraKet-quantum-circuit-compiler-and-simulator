"""Tests for the extended gate set (S/T/SX, RX/RY/P, CZ/CY/SWAP, aliases).

Covers parsing (incl. aliases), native decomposition (equal up to global phase,
output only native), and the new optimizer rules (inverse-pair cancellation and
RX/RY/P merging).
"""

import math

import pytest

from braket.ast_nodes import Gate
from braket.decomposer import NATIVE_GATES, decompose
from braket.optimizer import optimize
from braket.parser import parse
from tests.helpers import state_of, states_equal_up_to_global_phase, gate_summary


def is_native(circuit):
    return all(g.name in NATIVE_GATES for g in circuit.gates)


# --------------------------------------------------------------------------- #
# Parsing and aliases
# --------------------------------------------------------------------------- #

def test_parse_new_single_and_two_qubit_gates():
    src = ("qubits 2\nS q0\nT q1\nSX q0\nSDG q1\nRX(0.5) q0\nRY(0.6) q1\n"
           "P(0.7) q0\nCZ q0 q1\nCY q0 q1\nSWAP q0 q1\n")
    circuit = parse(src)
    assert [g.name for g in circuit.gates] == [
        "S", "T", "SX", "SDG", "RX", "RY", "P", "CZ", "CY", "SWAP"
    ]


def test_cx_alias_becomes_cnot():
    circuit = parse("qubits 2\nCX q0 q1\n")
    assert circuit.gates[0].name == "CNOT"


def test_identity_alias_normalizes():
    circuit = parse("qubits 1\nI q0\n")
    assert circuit.gates[0].name == "ID"


def test_new_param_gate_wrong_arity_still_errors():
    from braket.errors import ParseError
    with pytest.raises(ParseError):
        parse("qubits 2\nCZ q0\n")  # CZ needs two qubits


# --------------------------------------------------------------------------- #
# Native decomposition — equal up to global phase, output only native
# --------------------------------------------------------------------------- #

SINGLE_QUBIT_LINES = [
    "S q0", "SDG q0", "T q0", "TDG q0", "SX q0", "SXDG q0",
    "RX(0.7) q0", "RY(1.3) q0", "RZ(2.1) q0", "P(0.9) q0",
    "H q0", "X q0", "Y q0", "Z q0", "ID q0",
]


@pytest.mark.parametrize("line", SINGLE_QUBIT_LINES)
def test_single_qubit_gate_decomposes_to_native(line):
    # Prepare with H so relative phases matter in the comparison.
    original = parse(f"qubits 1\nH q0\n{line}\n")
    decomposed = decompose(original)
    assert is_native(decomposed)
    assert states_equal_up_to_global_phase(state_of(original), state_of(decomposed))


TWO_QUBIT_LINES = ["CNOT q0 q1", "CZ q0 q1", "CY q0 q1", "SWAP q0 q1"]


@pytest.mark.parametrize("line", TWO_QUBIT_LINES)
def test_two_qubit_gate_decomposes_to_native(line):
    original = parse(f"qubits 2\nH q0\nH q1\n{line}\n")
    decomposed = decompose(original)
    assert is_native(decomposed)
    assert states_equal_up_to_global_phase(state_of(original), state_of(decomposed))


# --------------------------------------------------------------------------- #
# Optimizer: inverse-pair cancellation and rotation merging
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("g,gdg", [("S", "SDG"), ("T", "TDG"), ("SX", "SXDG")])
def test_inverse_pairs_cancel(g, gdg):
    assert optimize(parse(f"qubits 1\n{g} q0\n{gdg} q0\n")).gates == []
    # ...and the reverse order too.
    assert optimize(parse(f"qubits 1\n{gdg} q0\n{g} q0\n")).gates == []


def test_same_phase_gate_does_not_cancel():
    # S;S == Z, not identity -> must NOT cancel.
    out = optimize(parse("qubits 1\nS q0\nS q0\n"))
    assert [g.name for g in out.gates] == ["S", "S"]


@pytest.mark.parametrize("rot", ["RX", "RY", "RZ", "P"])
def test_rotations_merge(rot):
    out = optimize(parse(f"qubits 1\n{rot}(0.4) q0\n{rot}(0.5) q0\n"))
    (only,) = out.gates
    assert only.name == rot
    assert math.isclose(only.params[0], 0.9)


def test_opposite_rotations_cancel_to_nothing():
    assert optimize(parse("qubits 1\nP(1.2) q0\nP(-1.2) q0\n")).gates == []


@pytest.mark.parametrize("twoq", ["CZ", "CY"])
def test_two_qubit_self_inverse_cancel(twoq):
    assert optimize(parse(f"qubits 2\n{twoq} q0 q1\n{twoq} q0 q1\n")).gates == []


def test_identity_gate_is_dropped():
    out = optimize(parse("qubits 1\nID q0\nX q0\n"))
    assert gate_summary(out) == [("X", (0,), ())]
