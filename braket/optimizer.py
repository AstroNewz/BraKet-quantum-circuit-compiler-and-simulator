"""Optimization pass (M3).

Rewrites a circuit into an equivalent, cheaper one using two peephole rules:

  (a) Self-inverse cancellation. Some gates are their own inverse: doing them
      twice in a row does nothing. So `H q0; H q0`, `X q0; X q0`, `CNOT q0 q1;
      CNOT q0 q1`, etc. cancel to nothing — provided *nothing else touches those
      qubits in between*. (A gate on unrelated qubits may sit between them; it
      commutes past and doesn't block the cancellation.)

  (b) RZ merging. Two Z-rotations on the same qubit combine into one:
      RZ(a); RZ(b) == RZ(a + b). If the summed angle is a multiple of 2*pi the
      rotation is the identity (up to an unobservable global phase) and is
      dropped entirely.

Both rules are *semantics preserving*: the optimized circuit computes the same
quantum state as the original (exactly, or up to global phase for the RZ-drop
case). The tests prove this by simulating before and after.

Design: we repeatedly apply one reduction and restart until no rule fires
(a fixpoint). Each reduction looks, for a gate at position i, at the *next*
instruction that touches any of its qubits — skipping over commuting gates on
other qubits, and stopping at a measurement (which acts as a barrier). This
"next toucher" idea is what makes the "nothing in between" condition precise.
"""

import math

from .ast_nodes import Circuit, Gate, Measure

# Gates that are their own inverse (G @ G == I). RZ is NOT here (it's handled by
# merging), and S/T/SX are NOT (their inverses are the "dg" variants below).
SELF_INVERSE_GATES = {"H", "X", "Y", "Z", "CNOT", "SWAP", "CZ", "CY"}

# Gates whose inverse is a *different* gate. Listed both ways so a lookup works
# from either direction. Doing G then G-inverse (with nothing between) cancels.
_INVERSE_PARTNERS = {
    "S": "SDG", "SDG": "S",
    "T": "TDG", "TDG": "T",
    "SX": "SXDG", "SXDG": "SX",
}

# Single-qubit rotation families whose adjacent gates on the same qubit merge by
# summing their angles: RX(a); RX(b) == RX(a+b), and likewise RY, RZ, P.
ROTATION_GATES = {"RX", "RY", "RZ", "P"}


def _inverse_name(name):
    """The gate name that cancels `name`, or None if it has no simple inverse."""
    if name in SELF_INVERSE_GATES:
        return name
    return _INVERSE_PARTNERS.get(name)


def optimize(circuit):
    """Return a new, equivalent `Circuit` with the two peephole rules applied."""
    instructions = list(circuit.instructions)

    changed = True
    while changed:
        changed = False
        for i, instruction in enumerate(instructions):
            if not isinstance(instruction, Gate):
                continue

            # Drop explicit identity gates outright.
            if instruction.name == "ID":
                del instructions[i]
                changed = True
                break

            j = _next_touching(instructions, i, instruction.qubits)
            if j is None:
                continue
            partner = instructions[j]
            if not isinstance(partner, Gate):
                continue  # a measurement barrier: can't cancel across it

            if _cancels(instruction, partner):
                # Remove the later one first so index i stays valid.
                del instructions[j]
                del instructions[i]
                changed = True
                break

            if instruction.name in ROTATION_GATES and partner.name == instruction.name \
                    and partner.qubits == instruction.qubits:
                net = instruction.params[0] + partner.params[0]
                del instructions[j]
                if _is_zero_mod_2pi(net):
                    del instructions[i]  # net rotation is identity -> drop both
                else:
                    instructions[i] = Gate(
                        instruction.name, instruction.qubits, (net,),
                        line=instruction.line, col=instruction.col,
                    )
                changed = True
                break

    return Circuit(circuit.n_qubits, instructions)


def _next_touching(instructions, i, qubits):
    """Index of the first instruction after `i` that touches any of `qubits`.

    Gates on disjoint qubits are skipped (they commute past). A measurement on one
    of `qubits` is returned as a barrier. Returns None if nothing else touches
    these qubits before the end of the circuit.
    """
    target = set(qubits)
    for j in range(i + 1, len(instructions)):
        instruction = instructions[j]
        if isinstance(instruction, Gate):
            if target & set(instruction.qubits):
                return j
        elif isinstance(instruction, Measure):
            if instruction.qubit in target:
                return j
    return None


def _cancels(first, second):
    """True if `second` undoes `first` on the same qubits.

    Covers self-inverse gates (H;H, CNOT;CNOT, CZ;CZ, ...) and inverse pairs
    (S;SDG, T;TDG, SX;SXDG). Qubit order must match exactly: `CNOT q0 q1` cancels
    a following `CNOT q0 q1`, but not `CNOT q1 q0` (a different operation).
    """
    return (
        second.name == _inverse_name(first.name)
        and second.qubits == first.qubits
    )


def _is_zero_mod_2pi(theta, tol=1e-9):
    """True if `theta` is (numerically) a whole multiple of 2*pi."""
    remainder = math.remainder(theta, 2 * math.pi)  # nearest-to-zero residue
    return abs(remainder) < tol
