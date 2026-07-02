"""Qubit mapping / routing (M4).

Real hardware doesn't let any two qubits interact. Here the topology is a
*linear nearest-neighbor chain*: physical qubit i can only share a two-qubit gate
with its neighbours i-1 and i+1. A circuit written against ideal (all-to-all)
connectivity may ask for a CNOT between qubits that aren't neighbours, so the
router must fix that up.

The idea
--------
We keep a running map between *logical* qubits (what the program talks about) and
*physical* qubits (wires on the chip). It starts as the identity: logical q sits
on physical q. Then we walk the circuit:

  * A single-qubit gate can run anywhere, so we just relabel it onto the physical
    wire its logical qubit currently occupies.
  * A two-qubit gate needs its operands adjacent. If they aren't, we insert SWAP
    gates (each swapping two neighbouring wires) to walk one operand along the
    chain until it sits next to the other, updating the map as we go, then emit
    the gate on the now-adjacent physical wires.

We don't swap qubits back afterwards — that would waste gates. Instead the map
simply evolves, and we remember the final layout so measurements can be read off
the correct wires. Because SWAP physically moves a qubit's state to another wire,
measuring the final wire of logical qubit q gives exactly q's outcome — so the
measurement statistics are identical to the original, just on relabelled wires.

This "walk one qubit over" strategy is deliberately simple (not an optimal router)
— the point is a correct, understandable pass, per the project's goals.
"""

from dataclasses import dataclass

from .ast_nodes import Circuit, Gate, Measure


@dataclass
class RoutingResult:
    """The routed circuit plus the qubit layouts before and after.

    `circuit`        — new circuit over physical qubits, with SWAPs inserted and
                       every two-qubit gate acting on adjacent wires.
    `initial_layout` — logical->physical map at the start (the identity).
    `final_layout`   — logical->physical map at the end; `final_layout[q]` is the
                       physical wire holding logical qubit q after routing. Use it
                       to interpret measurement results.
    """

    circuit: Circuit
    initial_layout: list
    final_layout: list


def are_adjacent(physical_a, physical_b):
    """Linear-chain connectivity: two wires can interact iff they're neighbours."""
    return abs(physical_a - physical_b) == 1


def route(circuit):
    """Route `circuit` onto the linear chain, returning a `RoutingResult`."""
    n = circuit.n_qubits

    # pos[logical] = physical wire it currently sits on.
    # occ[physical] = logical qubit currently on that wire (the inverse map).
    pos = list(range(n))
    occ = list(range(n))

    routed = []
    for instruction in circuit.instructions:
        if isinstance(instruction, Gate):
            if len(instruction.qubits) == 1:
                (logical,) = instruction.qubits
                routed.append(_relabel(instruction, (pos[logical],)))
            else:
                a, b = instruction.qubits
                routed.extend(_bring_adjacent(pos, occ, a, b))
                routed.append(_relabel(instruction, (pos[a], pos[b])))
        elif isinstance(instruction, Measure):
            # Measurements are deferred to the end (no mid-circuit measurement in
            # this version), so we place them below using the final layout.
            continue
        else:
            raise TypeError(f"cannot route instruction {instruction!r}")

    # Emit each requested measurement on the final wire of its logical qubit,
    # preserving the original measurement order.
    for logical in circuit.measured_qubits:
        routed.append(Measure(pos[logical]))

    return RoutingResult(
        circuit=Circuit(n, routed),
        initial_layout=list(range(n)),
        final_layout=list(pos),
    )


def _bring_adjacent(pos, occ, a, b):
    """Insert SWAPs (mutating `pos`/`occ`) until logical `a` and `b` are neighbours.

    Returns the list of SWAP gates emitted. We repeatedly move whichever operand
    sits lower on the chain one step up toward the other, shrinking the gap by one
    each time, until the two are adjacent.
    """
    swaps = []
    while not are_adjacent(pos[a], pos[b]):
        lower = a if pos[a] < pos[b] else b
        swaps.append(_swap_neighbours(pos, occ, pos[lower]))
    return swaps


def _swap_neighbours(pos, occ, physical):
    """Swap the wires `physical` and `physical + 1`; update the maps; return the gate."""
    p, q = physical, physical + 1
    logical_p, logical_q = occ[p], occ[q]

    occ[p], occ[q] = logical_q, logical_p
    pos[logical_p], pos[logical_q] = q, p

    return Gate("SWAP", (p, q))


def _relabel(gate, physical_qubits):
    """A copy of `gate` acting on the given physical qubits (params/loc preserved)."""
    return Gate(gate.name, tuple(physical_qubits), gate.params, line=gate.line, col=gate.col)
