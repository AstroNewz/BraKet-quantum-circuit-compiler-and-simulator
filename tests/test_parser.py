"""M1 parser tests.

Two halves:
  1. Valid circuits parse into the expected `Circuit` AST.
  2. Every documented error case raises a `ParseError` with a specific, located
     message — this is the "real compiler reports errors well" requirement.
"""

import math

import pytest

from braket.ast_nodes import Circuit, Gate, Measure
from braket.errors import ParseError
from braket.parser import parse


# --------------------------------------------------------------------------- #
# Valid circuits
# --------------------------------------------------------------------------- #

def test_parse_bell_circuit():
    source = """
    qubits 2
    H q0
    CNOT q0 q1
    MEASURE q0
    MEASURE q1
    """
    circuit = parse(source)
    assert isinstance(circuit, Circuit)
    assert circuit.n_qubits == 2
    assert circuit.instructions == [
        Gate("H", (0,), (), line=3, col=5),
        Gate("CNOT", (0, 1), (), line=4, col=5),
        Measure(0, line=5, col=5),
        Measure(1, line=6, col=5),
    ]


def test_parse_all_single_qubit_gates():
    circuit = parse("qubits 1\nH q0\nX q0\nY q0\nZ q0\n")
    assert [g.name for g in circuit.gates] == ["H", "X", "Y", "Z"]
    assert all(g.qubits == (0,) for g in circuit.gates)


def test_parse_rz_parameter_is_float():
    circuit = parse("qubits 1\nRZ(1.5708) q0\n")
    (gate,) = circuit.gates
    assert gate.name == "RZ"
    assert gate.qubits == (0,)
    assert len(gate.params) == 1
    assert math.isclose(gate.params[0], 1.5708)


def test_parse_negative_rz_parameter():
    circuit = parse("qubits 1\nRZ(-3.14) q0\n")
    assert math.isclose(circuit.gates[0].params[0], -3.14)


def test_measured_qubits_helper_dedupes_in_order():
    circuit = parse("qubits 3\nMEASURE q2\nMEASURE q0\nMEASURE q2\n")
    assert circuit.measured_qubits == [2, 0]


def test_empty_circuit_with_only_declaration_is_valid():
    circuit = parse("qubits 3\n")
    assert circuit.n_qubits == 3
    assert circuit.instructions == []


def test_comments_do_not_affect_parsing():
    source = "# bell\nqubits 2  # two qubits\nH q0\nCNOT q0 q1  # entangle\n"
    circuit = parse(source)
    assert [g.name for g in circuit.gates] == ["H", "CNOT"]


# --------------------------------------------------------------------------- #
# Error cases — each must be specific and located
# --------------------------------------------------------------------------- #

def _parse_error(source):
    with pytest.raises(ParseError) as exc:
        parse(source)
    return exc.value


def test_unknown_gate():
    err = _parse_error("qubits 2\nFOO q0\n")
    assert "unknown gate" in err.message
    assert "FOO" in err.message
    assert err.line == 2 and err.col == 1


def test_qubit_index_out_of_range():
    err = _parse_error("qubits 2\nH q5\n")
    assert "out of range" in err.message
    assert "q5" in err.message
    assert err.line == 2


def test_too_many_qubit_arguments():
    err = _parse_error("qubits 2\nH q0 q1\n")
    assert "expects 1 qubit" in err.message
    assert "got 2" in err.message


def test_too_few_qubit_arguments_for_cnot():
    err = _parse_error("qubits 2\nCNOT q0\n")
    assert "expects 2 qubits" in err.message
    assert "got 1" in err.message


def test_cnot_on_same_qubit():
    err = _parse_error("qubits 2\nCNOT q0 q0\n")
    assert "must be different" in err.message


def test_missing_register_declaration():
    err = _parse_error("H q0\n")
    # A gate before any 'qubits' declaration is the first thing we hit.
    assert "must be declared" in err.message


def test_no_qubits_declaration_at_all():
    err = _parse_error("# just a comment\n")
    assert "no 'qubits N' declaration" in err.message


def test_duplicate_register_declaration():
    err = _parse_error("qubits 2\nqubits 3\n")
    assert "already declared" in err.message
    assert err.line == 2


def test_non_integer_register_size():
    err = _parse_error("qubits 2.5\n")
    assert "positive whole number" in err.message


def test_zero_register_size():
    err = _parse_error("qubits 0\n")
    assert "positive whole number" in err.message


def test_rz_missing_parameter_parens():
    err = _parse_error("qubits 1\nRZ q0\n")
    assert "expected '('" in err.message


def test_rz_missing_closing_paren():
    err = _parse_error("qubits 1\nRZ(1.5 q0\n")
    assert "expected ')'" in err.message


def test_bad_qubit_reference_token():
    err = _parse_error("qubits 2\nH q0x\n")
    assert "expected a qubit like 'q0'" in err.message


def test_error_str_includes_caret_pointer():
    err = _parse_error("qubits 2\nFOO q0\n")
    rendered = str(err)
    assert "FOO q0" in rendered
    assert "^" in rendered
