"""M7 end-to-end tests: the shipped example circuits, through the whole pipeline.

For each example we run parse -> optimize -> route -> decompose and check that the
result is native, hardware-legal, and reproduces the exact textbook measurement
distribution. This is the "meaning is preserved across every stage" guarantee,
exercised on real example files rather than synthetic snippets.
"""

from pathlib import Path

import pytest

from braket.decomposer import NATIVE_GATES, decompose
from braket.mapper import are_adjacent, route
from braket.optimizer import optimize
from braket.parser import parse
from tests.helpers import assert_marginals_close, marginal_of

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# example file -> expected exact outcome distribution over all qubits (q0 first).
EXPECTED = {
    "bell_state.bkt": {"00": 0.5, "11": 0.5},
    "ghz_state.bkt": {"000": 0.5, "111": 0.5},
    "interference.bkt": {"1": 1.0},
}


def compile_all(circuit):
    routed = route(optimize(circuit))
    native = decompose(routed.circuit)
    return native, routed.final_layout


@pytest.mark.parametrize("filename, expected", EXPECTED.items())
def test_example_produces_expected_distribution(filename, expected):
    original = parse((EXAMPLES / filename).read_text())

    # The original (logical) circuit gives the textbook distribution.
    logical = list(range(original.n_qubits))
    assert_marginals_close(marginal_of(original, logical), expected)


@pytest.mark.parametrize("filename, expected", EXPECTED.items())
def test_example_compiles_to_native_and_preserves_meaning(filename, expected):
    original = parse((EXAMPLES / filename).read_text())
    native, layout = compile_all(original)

    # Native gate set only.
    for g in native.gates:
        assert g.name in NATIVE_GATES

    # Every two-qubit gate is on adjacent hardware wires.
    for g in native.gates:
        if len(g.qubits) == 2:
            assert are_adjacent(*g.qubits)

    # Reading each logical qubit off its final wire reproduces the expected result.
    wires = [layout[q] for q in range(original.n_qubits)]
    assert_marginals_close(marginal_of(native, wires), expected)


def test_all_example_files_parse():
    # Guard against a broken example slipping into the repo.
    for path in EXAMPLES.glob("*.bkt"):
        circuit = parse(path.read_text())
        assert circuit.n_qubits >= 1
