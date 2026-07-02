"""Native gate decomposition (M5).

Real hardware implements only a small "native" gate set. Here that set is:

    { RZ(theta), X90 = Rx(pi/2), CNOT }

The decomposer rewrites every other gate into a sequence of native gates. The
single-qubit gates H, X, Y, Z become RZ/X90 sequences; the SWAP gates that the
router (M4) inserts become three CNOTs, so a fully routed+decomposed circuit
contains *only* native gates.

Why these particular sequences work (the intuition)
---------------------------------------------------
Any single-qubit gate is some rotation of the Bloch sphere, and RZ (rotate about
Z) together with X90 (a quarter turn about X) generate all such rotations. The
specific constants below were derived and then checked numerically against the
exact matrices (see tests/test_decomposer.py), each holding *up to a global
phase* — an overall factor like i or -1 that multiplies the whole state and can
never be observed, so it is physically irrelevant.

    Z = RZ(pi)                          (RZ(pi) = -i*Z)
    X = X90 . X90                       (two quarter-turns about X = a half-turn)
    H = RZ(pi/2) . X90 . RZ(pi/2)       (a Z/X/Z Euler decomposition)
    Y = X . Z  = RZ(pi) then X90 . X90  (Y = i*X*Z)

    SWAP(a,b) = CNOT(a,b) . CNOT(b,a) . CNOT(a,b)   (exact, no phase)

Reminder on ordering: an instruction list is applied left to right (first
instruction acts first), so the sequence [RZ, X90, RZ] realises the operator
RZ . X90 . RZ.
"""

import math

from .ast_nodes import Circuit, Gate, Measure

# The target hardware alphabet. A decomposed circuit uses only these.
NATIVE_GATES = {"RZ", "X90", "CNOT"}

_PI = math.pi


def _rz(q, theta):
    return Gate("RZ", (q,), (theta,))


def _x90(q):
    return Gate("X90", (q,))


def _native_h(q):
    """H as native gates: RZ(pi/2) X90 RZ(pi/2) (up to global phase)."""
    return [_rz(q, _PI / 2), _x90(q), _rz(q, _PI / 2)]


# Each rule maps a gate name to a function of the whole `Gate` (so parametrized
# gates can read their angle) returning the native replacement sequence. All were
# verified numerically up to global phase (see tests/test_decomposer.py).
_SINGLE_QUBIT_RULES = {
    "ID": lambda g: [],
    "Z": lambda g: [_rz(g.qubits[0], _PI)],
    "X": lambda g: [_x90(g.qubits[0]), _x90(g.qubits[0])],
    "Y": lambda g: [_rz(g.qubits[0], _PI), _x90(g.qubits[0]), _x90(g.qubits[0])],
    "H": lambda g: _native_h(g.qubits[0]),
    "S": lambda g: [_rz(g.qubits[0], _PI / 2)],
    "SDG": lambda g: [_rz(g.qubits[0], -_PI / 2)],
    "T": lambda g: [_rz(g.qubits[0], _PI / 4)],
    "TDG": lambda g: [_rz(g.qubits[0], -_PI / 4)],
    "SX": lambda g: [_x90(g.qubits[0])],
    "SXDG": lambda g: [_x90(g.qubits[0]), _x90(g.qubits[0]), _x90(g.qubits[0])],
    "P": lambda g: [_rz(g.qubits[0], g.params[0])],
    # RX(t) = RZ(pi/2) X90 RZ(t+pi) X90 RZ(pi/2)  (a palindrome; order-agnostic)
    "RX": lambda g: [
        _rz(g.qubits[0], _PI / 2),
        _x90(g.qubits[0]),
        _rz(g.qubits[0], g.params[0] + _PI),
        _x90(g.qubits[0]),
        _rz(g.qubits[0], _PI / 2),
    ],
    # RY(t) = X90 RZ(t+pi) X90 RZ(pi)
    "RY": lambda g: [
        _x90(g.qubits[0]),
        _rz(g.qubits[0], g.params[0] + _PI),
        _x90(g.qubits[0]),
        _rz(g.qubits[0], _PI),
    ],
}


def decompose(circuit):
    """Return a new `Circuit` containing only native gates (plus measurements)."""
    out = []
    for instruction in circuit.instructions:
        if isinstance(instruction, Measure):
            out.append(instruction)
        else:
            out.extend(_decompose_gate(instruction))
    return Circuit(circuit.n_qubits, out)


def _decompose_gate(gate):
    name = gate.name

    # Already native: pass straight through.
    if name in NATIVE_GATES:
        return [gate]

    if name in _SINGLE_QUBIT_RULES:
        return _SINGLE_QUBIT_RULES[name](gate)

    if name == "SWAP":
        a, b = gate.qubits
        return [Gate("CNOT", (a, b)), Gate("CNOT", (b, a)), Gate("CNOT", (a, b))]

    if name == "CZ":
        # CZ = H(target) . CNOT . H(target); CZ is symmetric so pick qubit[1].
        control, target = gate.qubits
        return _native_h(target) + [Gate("CNOT", (control, target))] + _native_h(target)

    if name == "CY":
        # CY = S(target) . CNOT . SDG(target).
        control, target = gate.qubits
        return [
            _rz(target, -_PI / 2),
            Gate("CNOT", (control, target)),
            _rz(target, _PI / 2),
        ]

    raise ValueError(f"no native decomposition known for gate {name!r}")
