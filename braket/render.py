"""ASCII rendering helpers for the CLI: a banner, titled boxes, and — the fun one
— a text circuit diagram.

Everything here is plain 7-bit ASCII (no box-drawing Unicode), so it renders the
same on a Windows terminal as anywhere else.

The circuit diagram draws one horizontal wire per qubit and one column per
instruction, placing gate boxes on wires and drawing vertical connectors for
two-qubit gates. For example a Bell circuit renders as:

    q0: --[H]--@----[M]-------
               |
    q1: -------(+)-------[M]--
"""

from .ast_nodes import Gate, Measure


# --------------------------------------------------------------------------- #
# Banner and boxes
# --------------------------------------------------------------------------- #

def banner():
    """A thematic ASCII banner (uses Dirac's <bra|ket> notation)."""
    return "\n".join([
        "+==========================================================+",
        "|                                                          |",
        "|   BraKet  -  a quantum circuit compiler & simulator      |",
        "|                                                          |",
        "|                  <  bra | ket  >                         |",
        "|                                                          |",
        "+==========================================================+",
    ])


def box(title, body):
    """Wrap `body` (a string or list of lines) in a titled ASCII box.

        +-[ title ]-----------+
        | line one            |
        | line two            |
        +---------------------+
    """
    lines = body.splitlines() if isinstance(body, str) else list(body)
    if not lines:
        lines = [""]
    content_w = max([len(line) for line in lines] + [len(title) + 4])

    top = f"+-[ {title} ]" + "-" * (content_w - len(title) - 3) + "+"
    out = [top]
    for line in lines:
        out.append("| " + line.ljust(content_w) + " |")
    out.append("+" + "-" * (content_w + 2) + "+")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Circuit diagram
# --------------------------------------------------------------------------- #

def _two_qubit_symbols(name):
    """(symbol on first qubit, symbol on second qubit) for a two-qubit gate."""
    return {
        "CNOT": ("@", "(+)"),   # control, target
        "CZ": ("@", "@"),       # symmetric controlled-Z
        "CY": ("@", "[Y]"),     # control, Y target
        "SWAP": ("x", "x"),     # swap ends
    }.get(name, (f"[{name}]", f"[{name}]"))


def _center(text, width, fill):
    if len(text) >= width:
        return text
    pad = width - len(text)
    left = pad // 2
    return fill * left + text + fill * (pad - left)


def circuit_diagram(circuit):
    """Render `circuit` as a multi-line ASCII diagram."""
    n = circuit.n_qubits

    # Each instruction becomes one column: a map from qubit -> symbol, plus the
    # vertical span it covers (for connector lines).
    columns = []
    for instruction in circuit.instructions:
        cell = {}
        if isinstance(instruction, Measure):
            cell[instruction.qubit] = "[M]"
            span = (instruction.qubit, instruction.qubit)
        elif isinstance(instruction, Gate):
            if len(instruction.qubits) == 1:
                q = instruction.qubits[0]
                cell[q] = f"[{instruction.name}]"
                span = (q, q)
            else:
                a, b = instruction.qubits
                sym_a, sym_b = _two_qubit_symbols(instruction.name)
                cell[a], cell[b] = sym_a, sym_b
                span = (min(a, b), max(a, b))
        else:
            continue
        columns.append((cell, span))

    # Text grid: wire rows at even indices, connector "gap" rows at odd indices.
    total_rows = 2 * n - 1
    prefix_w = len(f"q{n - 1}:") + 1
    lines = []
    for r in range(total_rows):
        if r % 2 == 0:
            lines.append(f"q{r // 2}:".ljust(prefix_w) + "-")
        else:
            lines.append(" " * prefix_w + " ")

    for cell, (lo, hi) in columns:
        width = max((len(s) for s in cell.values()), default=1)
        for q in range(n):
            row = 2 * q
            symbol = cell.get(q)
            lines[row] += _center(symbol, width, "-") if symbol else "-" * width
        for gap in range(1, total_rows, 2):
            top_qubit = (gap - 1) // 2  # gap sits between top_qubit and top_qubit+1
            spanned = lo <= top_qubit and (top_qubit + 1) <= hi
            lines[gap] += _center("|", width, " ") if spanned else " " * width
        # one-character separator between columns
        for q in range(n):
            lines[2 * q] += "-"
        for gap in range(1, total_rows, 2):
            lines[gap] += " "

    return "\n".join(lines)
