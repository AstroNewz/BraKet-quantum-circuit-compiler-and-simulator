# BraKet

A small but *real* quantum circuit compiler and state-vector simulator, built for learning.

BraKet takes a circuit written in a tiny text format and runs it through a genuine compiler
pipeline before simulating it.

## Features

- **A real compiler pipeline**, not a black box: lexer → parser → optimizer → hardware router
  → native-gate decomposer → state-vector simulator, each in its own readable module.
- **Friendly, located error messages** with a caret pointing at the problem, like a real
  compiler (`line 2, col 1: unknown gate 'FOO'`).
- **A proper state-vector simulator** that applies any gate to any qubit(s) via tensor
  reshaping — works for any number of qubits, no hard-coded special cases.
- **Optimizations**: cancel self-inverse / inverse-pair gates and merge adjacent rotations.
- **Hardware-aware routing** onto a linear nearest-neighbor chain by inserting SWAPs.
- **Native-gate decomposition** to `{RZ, X90, CNOT}`, with every rewrite verified numerically.
- **A decorated ASCII CLI** that draws the circuit, shows every compilation stage, and plots a
  sampled-measurement histogram.
- **A broad gate set**: `H X Y Z S SDG T TDG SX SXDG ID`, rotations `RX RY RZ P`, and two-qubit
  `CNOT`/`CX` `CZ` `CY` `SWAP`.
- **197 tests** checking exact analytical results and semantics-preservation across every stage,
  including an optional cross-check against Qiskit.

## Pipeline

```
source text
   │  (lexer)      break text into tokens
   ▼
 tokens
   │  (parser)     build an AST: qubit count + a list of instructions
   ▼
  AST / Circuit
   │  (optimizer)  cancel redundant gates, merge rotations
   ▼
 optimized circuit
   │  (mapper)     insert SWAPs so 2-qubit gates act on adjacent hardware qubits
   ▼
 mapped circuit
   │  (decomposer) rewrite gates into a small "native" hardware gate set
   ▼
 native circuit
   │  (simulator)  evolve a 2^n complex state vector, then sample measurements
   ▼
 results: state vector, probabilities, sampled histogram
```

Each stage lives in its own module under `braket/` so you can read and understand them one
at a time. Correctness is checked against known analytical results (a Bell state gives the
textbook correlated 50/50 outcomes, etc.), not just "it didn't crash".

## The circuit language (`.bkt` files)

```
# comments start with '#'
qubits 2          # declare the register size

H q0              # single-qubit gate
CNOT q0 q1        # two-qubit gate
RZ(1.5708) q0     # parametrized rotation, angle in radians
MEASURE q0        # mark a qubit to sample at the end
MEASURE q1
```

### Supported gates

| Kind | Gates |
|------|-------|
| Single-qubit, no parameter | `H` `X` `Y` `Z` `S` `SDG` `T` `TDG` `SX` `SXDG` `ID` |
| Single-qubit, one angle (radians) | `RX(θ)` `RY(θ)` `RZ(θ)` `P(θ)` |
| Two-qubit | `CNOT` (alias `CX`) `CZ` `CY` `SWAP` |
| Measurement | `MEASURE qN` |

`SDG`/`TDG`/`SXDG` are the inverses of `S`/`T`/`SX`; `P(θ)` is the phase gate
`diag(1, e^{iθ})`. All of them compile down to the native set `{RZ, X90, CNOT}`.

## Install

Requires Python 3.11+ and numpy. From the `braket/` project directory:

```bash
pip install -e ".[dev]"          # installs BraKet plus pytest for the test suite
pip install -e ".[validation]"   # optional: qiskit, only for the M6 cross-check tests
```

## Run

```bash
braket run examples/bell_state.bkt --shots 1000
# or, without installing the script entry point:
python -m braket.cli run examples/bell_state.bkt --shots 1000
```

Flags: `--shots N` samples N measurements, `--seed S` makes sampling reproducible, and
`--no-optimize` / `--no-map` / `--no-decompose` turn individual compiler stages off so you can
see what each one does.

### Sample output

```
+-[ Circuit diagram (as written) ]-+
| q0: -[H]--@--[M]-----            |
|           |                      |
| q1: -----(+)-----[M]-            |
+----------------------------------+

+-[ Native circuit  {RZ, X90, CNOT} ]-+
| RZ(1.5708) q0                       |
| X90 q0                              |
| RZ(1.5708) q0                       |
| CNOT q0 q1                          |
| MEASURE q0                          |
| MEASURE q1                          |
+-------------------------------------+

+-[ Sampled measurements (1000 shots; qubits q0, q1) ]-----+
| 00  ######################################## 502 (50.2%) |
| 11  ######################################## 498 (49.8%) |
+----------------------------------------------------------+

+-[ Pipeline self-check ]--------------------------+
| compiled circuit matches original statistics: OK |
+--------------------------------------------------+
```

## Test

```bash
pytest
```

## Project status

Built milestone by milestone:

| Milestone | What it adds                                           | Status |
|-----------|--------------------------------------------------------|--------|
| M0        | Project scaffold, tooling, empty modules               | done   |
| M1        | Lexer + parser (+ friendly error messages)             | done   |
| M2        | State-vector simulator                                 | done   |
| M3        | Optimization pass                                      | done   |
| M4        | Qubit mapping / routing                                | done   |
| M5        | Native gate decomposition                              | done   |
| M6        | Cross-check against qiskit (optional)                  | done   |
| M7        | CLI + example circuits                                 | done   |

Later additions: a broadened gate set and a decorated ASCII CLI (circuit diagrams + boxed
output).

## License

Released under the [MIT License](LICENSE).
