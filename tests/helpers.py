"""Shared test helpers for checking that a circuit transform preserves meaning.

The compiler stages (optimizer, mapper, decomposer) change a circuit's
*representation*. To prove they don't change its *meaning*, we simulate the
circuit before and after and compare the resulting quantum states.

Two notions of "same state" matter:

* Exactly equal amplitudes — the strictest check.
* Equal up to a global phase — a state and (e^{i*phi}) times that state are
  physically identical; the overall phase can never be observed. Some legitimate
  optimizations (dropping an RZ that sums to 2*pi) and the whole idea of gate
  decomposition only hold up to global phase, so this is the right notion there.

Probabilities are invariant under global phase, so comparing probability vectors
is another clean, phase-agnostic equivalence check.
"""

import numpy as np

from braket.simulator import StateVector


def state_of(circuit):
    """Final amplitude vector produced by simulating `circuit`'s gates."""
    sv = StateVector(circuit.n_qubits)
    sv.run(circuit)
    return sv.state


def probabilities_of(circuit):
    sv = StateVector(circuit.n_qubits)
    sv.run(circuit)
    return sv.probabilities()


def states_equal_up_to_global_phase(a, b, atol=1e-8):
    """True if |a> and |b> are equal up to an overall phase factor.

    For normalized states this is exactly the condition |<a|b>| == 1: the overlap
    magnitude reaches 1 only when the vectors are parallel in Hilbert space.
    """
    overlap = abs(np.vdot(a, b))
    return bool(np.isclose(overlap, 1.0, atol=atol))


def marginal_of(circuit, qubits):
    """Exact measurement distribution over `qubits` for `circuit`, as {bits: prob}."""
    sv = StateVector(circuit.n_qubits)
    sv.run(circuit)
    return sv.marginal(qubits)


def assert_marginals_close(a, b, atol=1e-9):
    """Assert two {bitstring: probability} dicts agree (missing key == 0)."""
    keys = set(a) | set(b)
    for k in keys:
        assert abs(a.get(k, 0.0) - b.get(k, 0.0)) < atol, (
            f"marginal mismatch at {k!r}: {a.get(k, 0.0)} vs {b.get(k, 0.0)}"
        )


def gate_summary(circuit):
    """A comparable, line/col-agnostic view of the gates: (name, qubits, rounded params)."""
    return [
        (g.name, g.qubits, tuple(round(p, 9) for p in g.params))
        for g in circuit.gates
    ]
