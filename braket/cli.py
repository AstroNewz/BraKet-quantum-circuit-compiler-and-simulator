"""Command-line interface (M7).

Wires the whole pipeline together behind:

    braket run circuit.bkt --shots 1000 [--seed S]
                           [--no-optimize] [--no-map] [--no-decompose]

It parses the circuit, then runs it through the compiler stages, printing the
circuit at each step so you can *see* what every pass did. Finally it simulates
the circuit and reports the final state vector, the outcome probabilities, and —
if you asked for shots — a sampled measurement histogram.

As a self-check, it also confirms the fully compiled native circuit is
equivalent to the original (same measurement statistics, once you account for the
router's qubit relabeling). That way the printed histogram isn't just plausible —
the pipeline is demonstrably faithful.
"""

import argparse
import sys

import numpy as np

from .decomposer import decompose
from .errors import BraKetError
from .mapper import route
from .optimizer import optimize
from .parser import parse
from .render import banner, box, circuit_diagram
from .simulator import StateVector, simulate

_AMPLITUDE_TOL = 1e-10
_MAX_STATES_SHOWN = 64


def main(argv=None):
    """Entry point for the `braket` command. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="braket",
        description="Compile and simulate a BraKet (.bkt) quantum circuit.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run_parser = subcommands.add_parser(
        "run", help="compile and simulate a .bkt circuit file"
    )
    run_parser.add_argument("file", help="path to a .bkt circuit file")
    run_parser.add_argument(
        "--shots", type=int, default=0,
        help="number of measurement samples to draw (0 = skip the histogram)",
    )
    run_parser.add_argument(
        "--seed", type=int, default=None,
        help="random seed for reproducible sampling",
    )
    run_parser.add_argument("--no-optimize", action="store_true", help="skip the optimizer")
    run_parser.add_argument("--no-map", action="store_true", help="skip qubit routing")
    run_parser.add_argument("--no-decompose", action="store_true", help="skip native decomposition")

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    parser.error(f"unknown command {args.command!r}")  # argparse exits; unreachable


def _run(args):
    # --- read the source ------------------------------------------------------
    try:
        with open(args.file, "r", encoding="utf-8") as handle:
            source = handle.read()
    except OSError as exc:
        print(f"braket: cannot read {args.file!r}: {exc}", file=sys.stderr)
        return 1

    # --- compile --------------------------------------------------------------
    try:
        parsed = parse(source)
    except BraKetError as exc:
        # Located, caret-annotated error message — the whole point of M1.
        print(f"braket: parse error in {args.file}:\n{exc}", file=sys.stderr)
        return 1

    print(banner())
    print()
    print(box("Circuit diagram (as written)", circuit_diagram(parsed)))
    print()
    _print_circuit("Parsed circuit", parsed)

    work = parsed
    if not args.no_optimize:
        work = optimize(work)
        _print_circuit("Optimized circuit", work)

    final_layout = list(range(parsed.n_qubits))
    if not args.no_map:
        routing = route(work)
        work = routing.circuit
        final_layout = routing.final_layout
        body = [repr(i) for i in work.instructions] or ["(no instructions)"]
        body += ["", "logical -> physical: " + _format_layout(final_layout)]
        print(box("Mapped circuit (linear nearest-neighbor chain)", body))
        print()
    if not args.no_decompose:
        work = decompose(work)
        _print_circuit("Native circuit  {RZ, X90, CNOT}", work)

    compiled = work

    if compiled is not parsed:
        print(box("Compiled circuit diagram", circuit_diagram(compiled)))
        print()

    # --- simulate (logical view, i.e. what the algorithm computes) ------------
    result = simulate(parsed, shots=args.shots, seed=args.seed)
    _print_state_vector(result.state, parsed.n_qubits)
    _print_probabilities(result.probabilities, parsed.n_qubits)

    if args.shots:
        _print_histogram(result.counts, result.measured_qubits, args.shots)

    # --- verify the compiled circuit is faithful ------------------------------
    _print_equivalence_check(parsed, compiled, final_layout)
    return 0


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #

def _print_circuit(title, circuit):
    body = [repr(i) for i in circuit.instructions] or ["(no instructions)"]
    print(box(title, body))
    print()


def _format_layout(layout):
    return ", ".join(f"q{logical}->wire{phys}" for logical, phys in enumerate(layout))


def _ket(index, n):
    return "|" + format(index, f"0{n}b") + ">"


def _format_amplitude(z):
    return f"{z.real:+.4f}{z.imag:+.4f}i"


def _print_state_vector(state, n):
    body = []
    shown = 0
    for index, amplitude in enumerate(state):
        if abs(amplitude) <= _AMPLITUDE_TOL:
            continue
        if shown >= _MAX_STATES_SHOWN:
            body.append("... (more states omitted)")
            break
        body.append(f"{_ket(index, n)}: {_format_amplitude(amplitude)}")
        shown += 1
    print(box("Final state vector", body))
    print()


def _print_probabilities(probs, n):
    body = [
        f"{_ket(index, n)}: {p:.4f}"
        for index, p in enumerate(probs)
        if p > _AMPLITUDE_TOL
    ]
    print(box("Outcome probabilities", body))
    print()


def _print_histogram(counts, measured_qubits, shots):
    label = ", ".join(f"q{q}" for q in measured_qubits)
    if not counts:
        print(box(f"Sampled measurements ({shots} shots; qubits {label})",
                  ["(no outcomes)"]))
        print()
        return
    width = 40
    most = max(counts.values())
    body = []
    for key in sorted(counts):
        count = counts[key]
        bar = "#" * max(1, round(width * count / most))
        body.append(f"{key}  {bar} {count} ({100 * count / shots:.1f}%)")
    print(box(f"Sampled measurements ({shots} shots; qubits {label})", body))
    print()


def _print_equivalence_check(original, compiled, final_layout):
    """Confirm the compiled circuit reproduces the original's statistics."""
    logical = list(range(original.n_qubits))
    wires = [final_layout[q] for q in logical]

    original_sv = StateVector(original.n_qubits)
    original_sv.run(original)
    compiled_sv = StateVector(compiled.n_qubits)
    compiled_sv.run(compiled)

    original_marginal = original_sv.marginal(logical)
    compiled_marginal = compiled_sv.marginal(wires)

    keys = set(original_marginal) | set(compiled_marginal)
    ok = all(
        abs(original_marginal.get(k, 0.0) - compiled_marginal.get(k, 0.0)) < 1e-9
        for k in keys
    )
    mark = "OK" if ok else "FAILED"
    print(box("Pipeline self-check",
              [f"compiled circuit matches original statistics: {mark}"]))


if __name__ == "__main__":
    sys.exit(main())
