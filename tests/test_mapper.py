"""M4 router tests.

We check three things:
  1. Structure — after routing, every two-qubit gate acts on ADJACENT physical
     wires (the hardware constraint), and adjacent gates are left untouched.
  2. Bookkeeping — the final logical->physical layout is correct.
  3. Meaning — measurement statistics are unchanged once we read each logical
     qubit off the physical wire it ended up on.
"""

import pytest

from braket.ast_nodes import Gate, Measure
from braket.mapper import are_adjacent, route
from braket.optimizer import optimize
from braket.parser import parse
from tests.helpers import assert_marginals_close, marginal_of


def two_qubit_gates(circuit):
    return [g for g in circuit.gates if len(g.qubits) == 2]


def assert_all_two_qubit_gates_adjacent(circuit):
    for g in two_qubit_gates(circuit):
        a, b = g.qubits
        assert are_adjacent(a, b), f"{g!r} acts on non-adjacent wires {a},{b}"


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #

def test_already_adjacent_cnot_needs_no_swaps():
    result = route(parse("qubits 2\nH q0\nCNOT q0 q1\n"))
    assert [g.name for g in result.circuit.gates] == ["H", "CNOT"]
    assert result.final_layout == [0, 1]  # nothing moved


def test_nonadjacent_cnot_inserts_swaps_and_becomes_adjacent():
    result = route(parse("qubits 3\nCNOT q0 q2\n"))
    names = [g.name for g in result.circuit.gates]
    assert "SWAP" in names
    assert_all_two_qubit_gates_adjacent(result.circuit)
    # Concretely: SWAP(0,1) moves logical 0 up to wire 1, then CNOT(1,2).
    assert [(g.name, g.qubits) for g in result.circuit.gates] == [
        ("SWAP", (0, 1)),
        ("CNOT", (1, 2)),
    ]
    assert result.final_layout == [1, 0, 2]


def test_distance_three_cnot_inserts_two_swaps():
    result = route(parse("qubits 4\nCNOT q0 q3\n"))
    swaps = [g for g in result.circuit.gates if g.name == "SWAP"]
    assert len(swaps) == 2
    assert_all_two_qubit_gates_adjacent(result.circuit)


def test_single_qubit_gates_follow_their_logical_qubit():
    # After routing q0<->wire moves, a later gate on q0 must land on q0's new wire.
    result = route(parse("qubits 3\nCNOT q0 q2\nX q0\n"))
    assert_all_two_qubit_gates_adjacent(result.circuit)
    # q0 ended on wire 1 (see previous test), so X q0 becomes X on wire 1.
    x_gate = [g for g in result.circuit.gates if g.name == "X"][0]
    assert x_gate.qubits == (result.final_layout[0],) == (1,)


def test_initial_layout_is_identity():
    result = route(parse("qubits 4\nCNOT q0 q3\n"))
    assert result.initial_layout == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# Meaning — statistics preserved under relabeling
# --------------------------------------------------------------------------- #

ROUTING_CIRCUITS = [
    # GHZ built with a non-adjacent CNOT (q0->q2).
    "qubits 3\nH q0\nCNOT q0 q1\nCNOT q0 q2\n",
    # Long-range entangler across the whole chain.
    "qubits 4\nH q0\nCNOT q0 q3\n",
    # Several non-adjacent interactions plus single-qubit gates and rotations.
    "qubits 4\nH q0\nCNOT q0 q2\nRZ(0.7) q3\nCNOT q1 q3\nX q0\n",
]


@pytest.mark.parametrize("src", ROUTING_CIRCUITS)
def test_routing_preserves_measurement_statistics(src):
    original = parse(src)
    result = route(original)

    # Measure every logical qubit so we compare full statistics.
    logical_qubits = list(range(original.n_qubits))
    physical_wires = [result.final_layout[q] for q in logical_qubits]

    logical_marginal = marginal_of(original, logical_qubits)
    physical_marginal = marginal_of(result.circuit, physical_wires)

    # Reading each logical qubit off its final wire reproduces the exact same
    # outcome distribution.
    assert_marginals_close(logical_marginal, physical_marginal)


@pytest.mark.parametrize("src", ROUTING_CIRCUITS)
def test_routed_circuit_obeys_hardware_connectivity(src):
    result = route(parse(src))
    assert_all_two_qubit_gates_adjacent(result.circuit)


def test_measurements_placed_on_final_wires():
    src = "qubits 3\nH q0\nCNOT q0 q2\nMEASURE q0\nMEASURE q2\n"
    original = parse(src)
    result = route(original)

    measures = [i for i in result.circuit.instructions if isinstance(i, Measure)]
    # MEASURE q0 -> q0's final wire; MEASURE q2 -> q2's final wire, order preserved.
    assert [m.qubit for m in measures] == [
        result.final_layout[0],
        result.final_layout[2],
    ]


def test_router_composes_with_optimizer():
    # A realistic pipeline slice: optimize then route; statistics still match.
    src = "qubits 3\nH q0\nH q0\nH q0\nCNOT q0 q2\nX q1\nX q1\n"
    original = parse(src)
    result = route(optimize(original))
    assert_all_two_qubit_gates_adjacent(result.circuit)

    wires = [result.final_layout[q] for q in range(3)]
    assert_marginals_close(
        marginal_of(original, [0, 1, 2]),
        marginal_of(result.circuit, wires),
    )
