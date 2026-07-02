"""State-vector simulator (M2).

A quantum state on n qubits is a unit vector of 2^n complex *amplitudes*, one per
computational basis state |000..0>, |000..1>, ... The probability of observing a
given basis state on measurement is the squared magnitude of its amplitude
(the Born rule). Applying a gate multiplies the state by that gate's unitary.

Convention (used everywhere in BraKet): qubit q0 is the MOST significant bit, so
basis-state integer b = q0*2^(n-1) + q1*2^(n-2) + ... + q_{n-1}*2^0. Concretely,
for 2 qubits the amplitude order is |q0 q1> = 00, 01, 10, 11.

How gates are applied (the important trick)
-------------------------------------------
Naively you could build the full 2^n x 2^n matrix for each gate and multiply. But
that matrix has 4^n entries — hopeless past ~12 qubits. Instead we *reshape* the
length-2^n vector into an n-dimensional tensor of shape (2, 2, ..., 2), where axis
i is qubit i, and contract the small gate matrix against just the axes it touches.
This uses only O(2^n) memory and works for ANY number of qubits and ANY target
qubits (adjacent or not) with no special cases — exactly what the brief asks for.
"""

from dataclasses import dataclass

import numpy as np

from . import gates
from .ast_nodes import Circuit, Gate, Measure


def _apply_single(state_tensor, matrix, target):
    """Apply a 2x2 `matrix` to `target` axis of the reshaped state tensor.

    tensordot contracts the matrix's input index (axis 1) with the target qubit
    axis; the matrix's output index becomes a new axis 0, which we move back into
    the target position so the axis-order still means "axis i == qubit i".
    """
    contracted = np.tensordot(matrix, state_tensor, axes=(1, target))
    return np.moveaxis(contracted, 0, target)


def _apply_pair(state_tensor, matrix4, qubit_a, qubit_b):
    """Apply a 4x4 `matrix4` to axes (`qubit_a`, `qubit_b`) of the state tensor.

    We reshape the 4x4 into a (2,2,2,2) tensor indexed [a_out, b_out, a_in, b_in],
    contract its two input axes against the two qubit axes, then move the two
    output axes back to positions a and b. Works for any pair, adjacent or not.
    """
    gate_tensor = matrix4.reshape(2, 2, 2, 2)
    contracted = np.tensordot(gate_tensor, state_tensor, axes=([2, 3], [qubit_a, qubit_b]))
    return np.moveaxis(contracted, [0, 1], [qubit_a, qubit_b])


class StateVector:
    """A mutable n-qubit state initialised to |00..0>, evolved by applying gates."""

    def __init__(self, n_qubits):
        self.n = n_qubits
        self.state = np.zeros(2 ** n_qubits, dtype=complex)
        self.state[0] = 1.0  # start in |00..0>

    # ---- evolution ------------------------------------------------------------

    def apply(self, name, qubits, params=()):
        """Apply gate `name` (with `params`) to the given `qubits` (a tuple)."""
        matrix = gates.matrix_for(name, params)
        tensor = self.state.reshape([2] * self.n)
        if len(qubits) == 1:
            tensor = _apply_single(tensor, matrix, qubits[0])
        elif len(qubits) == 2:
            tensor = _apply_pair(tensor, matrix, qubits[0], qubits[1])
        else:
            raise ValueError(f"gate {name!r} acts on {len(qubits)} qubits; only 1 or 2 supported")
        self.state = tensor.reshape(-1)
        return self

    def apply_gate(self, gate):
        """Apply a `Gate` AST node."""
        return self.apply(gate.name, gate.qubits, gate.params)

    def run(self, instructions):
        """Apply every `Gate` in a sequence (or `Circuit`); ignore measurements."""
        if isinstance(instructions, Circuit):
            instructions = instructions.instructions
        for ins in instructions:
            if isinstance(ins, Gate):
                self.apply_gate(ins)
            elif isinstance(ins, Measure):
                continue  # measurements are handled at the end, not mid-circuit
            else:
                raise TypeError(f"cannot simulate instruction {ins!r}")
        return self

    # ---- readout --------------------------------------------------------------

    def probabilities(self):
        """Probability of each basis state (length 2^n, sums to 1).

        Tiny negative values from floating-point round-off are clipped to 0.
        """
        probs = np.abs(self.state) ** 2
        return np.clip(probs.real, 0.0, None)

    def marginal(self, qubits):
        """Marginal outcome distribution over `qubits`, as {bitstring: probability}.

        This is the *exact* distribution (no sampling noise), handy for tests that
        assert a Bell state is precisely 50/50. `qubits` are read out in the given
        order, so qubits=[0,1] yields keys like "01" with q0 first.
        """
        probs = self.probabilities()
        distribution = {}
        for basis_index, p in enumerate(probs):
            if p == 0.0:
                continue
            key = self._bits_for(basis_index, qubits)
            distribution[key] = distribution.get(key, 0.0) + float(p)
        return distribution

    def sample(self, qubits, shots, seed=None):
        """Sample `shots` measurement outcomes over `qubits`.

        Returns {bitstring: count}. Uses numpy's PRNG, seedable for reproducible
        tests. Drawing from the full 2^n distribution and then projecting onto the
        requested qubits reproduces the correct marginal statistics, including the
        correlations that make e.g. a Bell state come out '00'/'11' only.
        """
        rng = np.random.default_rng(seed)
        probs = self.probabilities()
        probs = probs / probs.sum()  # renormalise against round-off before sampling
        draws = rng.choice(len(probs), size=shots, p=probs)

        counts = {}
        for basis_index in draws:
            key = self._bits_for(int(basis_index), qubits)
            counts[key] = counts.get(key, 0) + 1
        return counts

    # ---- helpers --------------------------------------------------------------

    def _bits_for(self, basis_index, qubits):
        """Extract the bits of `basis_index` for `qubits`, as a string (q0 = MSB)."""
        return "".join(str((basis_index >> (self.n - 1 - q)) & 1) for q in qubits)

    def __repr__(self):
        return f"StateVector(n={self.n}, state={np.array2string(self.state, precision=4)})"


@dataclass
class SimulationResult:
    """Everything the CLI/tests want from a run of a circuit."""

    n_qubits: int
    state: np.ndarray                 # final amplitudes, length 2^n
    measured_qubits: list             # which qubits were sampled (in order)
    probabilities: np.ndarray         # full 2^n probability vector
    counts: dict                      # {bitstring: shot count}, empty if shots == 0


def simulate(circuit, shots=0, seed=None):
    """Run a `Circuit` end to end and return a `SimulationResult`.

    If the circuit declares no MEASURE instructions we sample every qubit (q0..),
    so `--shots` always produces a meaningful histogram.
    """
    sv = StateVector(circuit.n_qubits)
    sv.run(circuit)

    measured = circuit.measured_qubits or list(range(circuit.n_qubits))
    counts = sv.sample(measured, shots, seed=seed) if shots else {}

    return SimulationResult(
        n_qubits=circuit.n_qubits,
        state=sv.state,
        measured_qubits=measured,
        probabilities=sv.probabilities(),
        counts=counts,
    )
