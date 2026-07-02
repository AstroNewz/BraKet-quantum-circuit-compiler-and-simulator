"""AST node definitions (M1).

The parser produces a `Circuit`: the declared qubit count plus an ordered list of
instructions. Instructions are either a `Gate` (a unitary applied to one or more
qubits, optionally with numeric parameters) or a `Measure` (mark a qubit to be
sampled at the end).

These are plain data objects with no behaviour, so every later pipeline stage
(optimizer, mapper, decomposer, simulator) can read and rewrite them freely.

`GATE_SIGNATURES` is the single source of truth for how many qubits and
parameters each gate takes. The parser uses it to validate source; later stages
reuse it when they synthesize gates (SWAP for routing, X90 for decomposition).
"""

from dataclasses import dataclass, field


# gate name -> (number of qubits it acts on, number of numeric parameters)
#
# Single-qubit, no parameters:
#   H, X, Y, Z          Hadamard and the Paulis
#   S, SDG              sqrt(Z) and its inverse (phase pi/2)
#   T, TDG              sqrt(S) and its inverse (phase pi/4)
#   SX, SXDG            sqrt(X) and its inverse (SX == the native X90)
#   I / ID              the identity (a no-op)
#   X90 = Rx(pi/2)      native 90-degree X rotation (used by the decomposer)
# Single-qubit, one angle parameter (radians):
#   RX, RY, RZ          rotations about the X/Y/Z axes
#   P                   phase gate diag(1, e^{i theta})
# Two-qubit, no parameters:
#   CNOT                controlled-NOT (control, target)
#   CZ                  controlled-Z (symmetric)
#   CY                  controlled-Y (control, target)
#   SWAP                exchange two qubits (also inserted by the mapper)
GATE_SIGNATURES = {
    "I": (1, 0),
    "ID": (1, 0),
    "H": (1, 0),
    "X": (1, 0),
    "Y": (1, 0),
    "Z": (1, 0),
    "S": (1, 0),
    "SDG": (1, 0),
    "T": (1, 0),
    "TDG": (1, 0),
    "SX": (1, 0),
    "SXDG": (1, 0),
    "X90": (1, 0),
    "RX": (1, 1),
    "RY": (1, 1),
    "RZ": (1, 1),
    "P": (1, 1),
    "CNOT": (2, 0),
    "CZ": (2, 0),
    "CY": (2, 0),
    "SWAP": (2, 0),
}

# Alternate spellings a user may write, mapped to the canonical gate name.
GATE_ALIASES = {
    "CX": "CNOT",   # CX is the common name for controlled-NOT
    "I": "ID",      # normalise the identity to a single internal name
}

# The subset a user is allowed to write in a .bkt source file (plus the aliases
# above). X90 is compiler-internal only, so the parser must not accept it.
SOURCE_GATES = {
    "H", "X", "Y", "Z", "S", "SDG", "T", "TDG", "SX", "SXDG", "ID", "I",
    "RX", "RY", "RZ", "P",
    "CNOT", "CX", "CZ", "CY", "SWAP",
}


@dataclass
class Gate:
    """A gate application, e.g. `H q0`, `RZ(1.57) q0`, or `CNOT q0 q1`.

    Attributes
    ----------
    name:    gate name, a key of GATE_SIGNATURES (e.g. "H", "CNOT").
    qubits:  the qubit indices it acts on, in order (control before target for CNOT).
    params:  numeric parameters (radians for RZ); empty for non-parametrized gates.
    line, col: source location of the gate keyword, for error messages (0 if synthesized).
    """

    name: str
    qubits: tuple
    params: tuple = ()
    line: int = 0
    col: int = 0

    def __repr__(self):
        qs = " ".join(f"q{q}" for q in self.qubits)
        if self.params:
            ps = ",".join(_fmt(p) for p in self.params)
            return f"{self.name}({ps}) {qs}"
        return f"{self.name} {qs}"


@dataclass
class Measure:
    """A measurement instruction: mark `qubit` to be sampled at the end."""

    qubit: int
    line: int = 0
    col: int = 0

    def __repr__(self):
        return f"MEASURE q{self.qubit}"


@dataclass
class Circuit:
    """A parsed circuit: register size plus an ordered instruction list."""

    n_qubits: int
    instructions: list = field(default_factory=list)

    @property
    def gates(self):
        """Just the `Gate` instructions, in order (measurements filtered out)."""
        return [ins for ins in self.instructions if isinstance(ins, Gate)]

    @property
    def measured_qubits(self):
        """Qubit indices marked with MEASURE, in first-seen order (deduplicated)."""
        seen = []
        for ins in self.instructions:
            if isinstance(ins, Measure) and ins.qubit not in seen:
                seen.append(ins.qubit)
        return seen

    def __repr__(self):
        body = "\n".join(f"  {ins!r}" for ins in self.instructions)
        return f"Circuit(n_qubits={self.n_qubits})\n{body}" if body else \
            f"Circuit(n_qubits={self.n_qubits})"


def _fmt(x):
    """Format a float compactly (trim trailing zeros) for readable repr output."""
    return f"{x:g}"
