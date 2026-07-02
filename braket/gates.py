"""Gate matrices (M2).

The unitary matrices for every gate BraKet knows about, as numpy arrays. Single-
qubit gates are 2x2; two-qubit gates are 4x4. Nothing here knows about qubit
*positions* in a register — the simulator is responsible for expanding a matrix
to act on the right qubit(s) of an n-qubit state.

A quick tour of the physics, since this is a learning project:

* X, Y, Z are the Pauli gates — the three "bit/phase flip" operations.
    X = NOT: |0> <-> |1>.
    Z: leaves |0>, flips the sign of |1> (a phase flip).
    Y: a combined bit+phase flip (Y = iXZ).
* H (Hadamard) creates superposition: |0> -> (|0>+|1>)/sqrt(2) = |+>.
* RZ(theta) rotates about the Z axis of the Bloch sphere; it multiplies |0> and
  |1> by opposite half-angle phases. It's the parametrized gate in our language.
* X90 = Rx(pi/2), a 90-degree rotation about X. It's part of the *native* gate set
  used in M5 — most hardware can't do an arbitrary gate directly, but it can do
  X90 and RZ, and those two are enough to build any single-qubit gate.
* CNOT flips the target qubit iff the control qubit is |1> — the standard way to
  create entanglement.
* SWAP exchanges two qubits' states; the mapper (M4) inserts these to move qubits
  next to each other on hardware that only allows neighbours to interact.

Two-qubit matrices use the convention that the *first* listed qubit is the more
significant bit (basis order |q_a q_b> = 00, 01, 10, 11), matching the simulator's
"q0 is most significant" rule.
"""

import numpy as np

# --- single-qubit constant gates ------------------------------------------- #

I2 = np.array([[1, 0], [0, 1]], dtype=complex)

H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

X = np.array([[0, 1], [1, 0]], dtype=complex)

Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

Z = np.array([[1, 0], [0, -1]], dtype=complex)

# X90 = Rx(pi/2) = 1/sqrt(2) * [[1, -i], [-i, 1]].
# Applying it twice gives Rx(pi) = -iX, i.e. an X up to global phase (see M5).
X90 = (1 / np.sqrt(2)) * np.array([[1, -1j], [-1j, 1]], dtype=complex)


def rz(theta):
    """RZ(theta) = diag(e^{-i theta/2}, e^{+i theta/2}).

    Rotation by angle `theta` (radians) about the Bloch-sphere Z axis. The two
    computational basis states pick up opposite half-angle phases; the difference
    between them (the *relative* phase) is what's physically meaningful.
    """
    return np.array(
        [[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]], dtype=complex
    )


def rx(theta):
    """RX(theta) = rotation by `theta` about the X axis = cos(t/2) I - i sin(t/2) X."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta):
    """RY(theta) = rotation by `theta` about the Y axis = cos(t/2) I - i sin(t/2) Y."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def p(theta):
    """P(theta) (phase gate) = diag(1, e^{i theta}).

    Leaves |0> alone and multiplies |1> by e^{i theta}. Equal to RZ(theta) up to
    an unobservable global phase. S, T and Z are the special cases theta =
    pi/2, pi/4, pi.
    """
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


# --- more single-qubit constant gates -------------------------------------- #

# Phase gates. S = sqrt(Z), T = sqrt(S); the "dg" variants are their inverses.
S = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
TDG = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)

# SX = sqrt(X) is exactly our native X90; SXDG is its inverse.
SX = X90
SXDG = X90.conj().T

# The identity (a no-op gate), occasionally handy as an explicit placeholder.
ID = I2


# --- two-qubit constant gates ---------------------------------------------- #
# Basis order for both is |first second> = |00>, |01>, |10>, |11>.

# CNOT with the first qubit as control: flips the second qubit when the first is 1
# (so it swaps the amplitudes of |10> and |11>).
CNOT = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=complex,
)

# SWAP exchanges the two qubits (swaps the amplitudes of |01> and |10>).
SWAP = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=complex,
)

# CZ flips the sign of |11> only (controlled-Z). Symmetric in its two qubits.
CZ = np.diag([1, 1, 1, -1]).astype(complex)

# CY applies Y to the second qubit when the first is 1 (controlled-Y).
CY = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, 1j, 0],
    ],
    dtype=complex,
)


# Gates with no numeric parameters, looked up by name.
_CONSTANT_GATES = {
    "I": I2,
    "ID": ID,
    "H": H,
    "X": X,
    "Y": Y,
    "Z": Z,
    "S": S,
    "SDG": SDG,
    "T": T,
    "TDG": TDG,
    "SX": SX,
    "SXDG": SXDG,
    "X90": X90,
    "CNOT": CNOT,
    "SWAP": SWAP,
    "CZ": CZ,
    "CY": CY,
}

# Gates that take a single angle parameter, looked up by name.
_PARAM_GATES = {
    "RX": rx,
    "RY": ry,
    "RZ": rz,
    "P": p,
}


def matrix_for(name, params=()):
    """Return the unitary matrix for gate `name` with the given `params`.

    Single-qubit gates return a 2x2 array; two-qubit gates return a 4x4 array.
    Raises `ValueError` for an unknown gate name (a programming error at this
    point — the parser has already rejected unknown *source* gates).
    """
    if name in _PARAM_GATES:
        (theta,) = params
        return _PARAM_GATES[name](theta)
    try:
        return _CONSTANT_GATES[name]
    except KeyError:
        raise ValueError(f"no matrix defined for gate {name!r}")
