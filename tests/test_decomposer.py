"""M5 decomposer tests.

We prove two properties:
  1. Correctness — a decomposed gate/circuit computes the same state as the
     original, up to an unobservable global phase.
  2. Nativeness — the output contains ONLY gates from {RZ, X90, CNOT}.

We also run the full pipeline (parse -> optimize -> route -> decompose) to confirm
all five stages compose into something that is both hardware-native and
statistics-preserving.
"""

import math

import pytest

from braket.ast_nodes import Circuit, Gate, Measure
from braket.decomposer import NATIVE_GATES, decompose
from braket.mapper import are_adjacent, route
from braket.optimizer import optimize
from braket.parser import parse
from tests.helpers import (
    assert_marginals_close,
    marginal_of,
    state_of,
    states_equal_up_to_global_phase,
)


def assert_native(circuit):
    for g in circuit.gates:
        assert g.name in NATIVE_GATES, f"non-native gate survived: {g!r}"


# --------------------------------------------------------------------------- #
# Per-gate correctness (up to global phase) and nativeness
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("gate", ["H", "X", "Y", "Z"])
def test_single_gate_decomposes_correctly(gate):
    original = parse(f"qubits 1\n{gate} q0\n")
    decomposed = decompose(original)

    assert_native(decomposed)
    assert states_equal_up_to_global_phase(state_of(original), state_of(decomposed))


def test_native_gates_pass_through_unchanged():
    # RZ and CNOT are already native and must be left exactly as-is.
    circuit = Circuit(2, [Gate("RZ", (0,), (0.9,)), Gate("CNOT", (0, 1))])
    out = decompose(circuit)
    assert [(g.name, g.qubits, g.params) for g in out.gates] == [
        ("RZ", (0,), (0.9,)),
        ("CNOT", (0, 1), ()),
    ]


def test_x90_passes_through_unchanged():
    circuit = Circuit(1, [Gate("X90", (0,))])
    out = decompose(circuit)
    assert [g.name for g in out.gates] == ["X90"]


def test_swap_decomposes_into_three_cnots_exactly():
    # Prepare with X90 (already native, so it is untouched by decompose); this
    # isolates the SWAP so we can check the SWAP rewrite is EXACT (no phase).
    circuit = Circuit(2, [Gate("X90", (0,)), Gate("SWAP", (0, 1))])
    out = decompose(circuit)
    cnots = [g for g in out.gates if g.name == "CNOT"]
    assert len(cnots) == 3
    assert [g.qubits for g in cnots] == [(0, 1), (1, 0), (0, 1)]
    assert_native(out)
    # SWAP -> 3 CNOTs is an exact identity, so states match with no global phase.
    import numpy as np
    assert np.allclose(state_of(circuit), state_of(out))


def test_measurements_are_preserved():
    original = parse("qubits 1\nH q0\nMEASURE q0\n")
    out = decompose(original)
    assert any(isinstance(i, Measure) for i in out.instructions)


# --------------------------------------------------------------------------- #
# Whole-circuit correctness
# --------------------------------------------------------------------------- #

DECOMPOSE_CIRCUITS = [
    "qubits 1\nH q0\nX q0\nY q0\nZ q0\n",
    "qubits 2\nH q0\nCNOT q0 q1\n",                 # Bell
    "qubits 3\nH q0\nCNOT q0 q1\nCNOT q1 q2\n",     # GHZ
    "qubits 2\nX q0\nRZ(0.6) q0\nH q1\nCNOT q0 q1\n",
]


@pytest.mark.parametrize("src", DECOMPOSE_CIRCUITS)
def test_decompose_preserves_state_up_to_global_phase(src):
    original = parse(src)
    decomposed = decompose(original)
    assert_native(decomposed)
    assert states_equal_up_to_global_phase(state_of(original), state_of(decomposed))


# --------------------------------------------------------------------------- #
# Full pipeline: parse -> optimize -> route -> decompose
# --------------------------------------------------------------------------- #

PIPELINE_CIRCUITS = [
    "qubits 3\nH q0\nCNOT q0 q1\nCNOT q0 q2\n",     # GHZ needing a route
    "qubits 4\nH q0\nCNOT q0 q3\nX q1\nX q1\n",     # long-range + cancelling pair
    "qubits 4\nH q0\nCNOT q0 q2\nRZ(0.4) q3\nCNOT q1 q3\n",
]


@pytest.mark.parametrize("src", PIPELINE_CIRCUITS)
def test_full_pipeline_is_native_adjacent_and_preserves_statistics(src):
    original = parse(src)

    routed = route(optimize(original))
    compiled = decompose(routed.circuit)

    # 1. Only native gates remain.
    assert_native(compiled)

    # 2. Every two-qubit (CNOT) gate acts on adjacent hardware wires.
    for g in compiled.gates:
        if len(g.qubits) == 2:
            assert are_adjacent(*g.qubits)

    # 3. Measurement statistics survive, read off each logical qubit's final wire.
    wires = [routed.final_layout[q] for q in range(original.n_qubits)]
    assert_marginals_close(
        marginal_of(original, list(range(original.n_qubits))),
        marginal_of(compiled, wires),
    )
