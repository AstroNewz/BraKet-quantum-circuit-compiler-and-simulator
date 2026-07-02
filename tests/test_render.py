"""Tests for the ASCII rendering helpers (banner, box, circuit diagram)."""

from braket.parser import parse
from braket.render import banner, box, circuit_diagram


def test_banner_mentions_braket():
    assert "BraKet" in banner()


def test_box_has_title_and_borders():
    rendered = box("Hello", ["line one", "a longer line two"])
    lines = rendered.splitlines()
    assert lines[0].startswith("+-[ Hello ]")
    assert lines[0].endswith("+")
    assert lines[-1].startswith("+") and lines[-1].endswith("+")
    # Every content line is wrapped in "| ... |".
    for line in lines[1:-1]:
        assert line.startswith("| ") and line.endswith(" |")
    # All lines share the same width (a well-formed box).
    assert len({len(line) for line in lines}) == 1


def test_box_handles_empty_body():
    rendered = box("Empty", [])
    assert "Empty" in rendered
    assert rendered.splitlines()[0].startswith("+-[ Empty ]")


def test_diagram_row_count_is_2n_minus_1():
    diagram = circuit_diagram(parse("qubits 3\nH q0\nCNOT q0 q1\nCNOT q1 q2\n"))
    assert len(diagram.splitlines()) == 2 * 3 - 1


def test_diagram_single_qubit_is_one_row():
    diagram = circuit_diagram(parse("qubits 1\nH q0\nMEASURE q0\n"))
    lines = diagram.splitlines()
    assert len(lines) == 1
    assert "[H]" in lines[0]
    assert "[M]" in lines[0]


def test_bell_diagram_has_expected_symbols():
    diagram = circuit_diagram(parse("qubits 2\nH q0\nCNOT q0 q1\nMEASURE q0\nMEASURE q1\n"))
    # Wire labels.
    assert "q0:" in diagram and "q1:" in diagram
    # Hadamard box, CNOT control '@' and target '(+)', and a vertical connector.
    assert "[H]" in diagram
    assert "@" in diagram
    assert "(+)" in diagram
    assert "|" in diagram  # connector between control and target
    assert "[M]" in diagram


def test_two_qubit_gate_symbols_render():
    for gate, symbol in [("CZ", "@"), ("SWAP", "x")]:
        diagram = circuit_diagram(parse(f"qubits 2\n{gate} q0 q1\n"))
        assert symbol in diagram
