"""Parser: tokens -> Circuit AST (M1).

Reads the flat token list from the lexer and builds a validated `Circuit`. This
is where we enforce:

  * grammar        — each statement has the right shape (`qubits N`, `H q0`,
                     `RZ(theta) qN`, `CNOT qA qB`, `MEASURE qN`);
  * static semantics we can check without simulating — qubit indices are in
    range, argument counts are correct, the register is declared exactly once
    before any gate, and CNOT's two qubits differ.

Anything wrong raises a `ParseError` carrying the source location and line, so
the message can point a caret at the problem.

Grammar (informally):

    program   := statement*
    statement := qubits_decl | gate | measure
    qubits_decl := "qubits" INT
    gate      := ("H"|"X"|"Y"|"Z") qubit
               | "RZ" "(" NUMBER ")" qubit
               | "CNOT" qubit qubit
    measure   := "MEASURE" qubit
    qubit     := IDENT matching  q<non-negative integer>

Statements are separated by NEWLINE tokens; blank and comment-only lines produce
empty statements that are simply skipped.
"""

import re

from .ast_nodes import (
    GATE_ALIASES,
    GATE_SIGNATURES,
    SOURCE_GATES,
    Circuit,
    Gate,
    Measure,
)
from .errors import ParseError
from .lexer import tokenize
from .tokens import TokenKind

# A qubit reference is the letter 'q' followed by one or more digits: q0, q12.
_QUBIT_RE = re.compile(r"^q(\d+)$")


def parse(source):
    """Parse circuit `source` text into a `Circuit`. Raises `ParseError`/`LexError`."""
    tokens = tokenize(source)
    return _Parser(tokens, source).parse_program()


class _Parser:
    """A small hand-written recursive-descent parser over the token list.

    Hand-written (rather than a parser-generator) on purpose: the grammar is tiny
    and this keeps every decision — and every error message — visible and easy to
    follow, which is the whole point of the project.
    """

    def __init__(self, tokens, source):
        self.tokens = tokens
        self.pos = 0
        # Keep the raw source lines so errors can show the offending line + caret.
        self.lines = source.split("\n")
        self.n_qubits = None          # set by the `qubits N` declaration
        self.instructions = []

    # ---- token cursor helpers -------------------------------------------------

    def _peek(self):
        return self.tokens[self.pos]

    def _advance(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _error(self, message, tok):
        """Raise a ParseError located at `tok`, including its source line."""
        source_line = self.lines[tok.line - 1] if 1 <= tok.line <= len(self.lines) else None
        raise ParseError(message, line=tok.line, col=tok.col, source_line=source_line)

    # ---- program / statement level -------------------------------------------

    def parse_program(self):
        while self._peek().kind != TokenKind.EOF:
            # Skip blank / comment-only lines (runs of NEWLINE).
            if self._peek().kind == TokenKind.NEWLINE:
                self._advance()
                continue
            self._parse_statement()
            self._expect_end_of_statement()

        if self.n_qubits is None:
            # Point at the very end of input; there's no specific token to blame.
            last = self.tokens[-1]
            raise ParseError(
                "no 'qubits N' declaration found; every circuit must declare its "
                "register size before any gates",
                line=last.line,
            )
        return Circuit(self.n_qubits, self.instructions)

    def _expect_end_of_statement(self):
        """After a full statement we must be at a NEWLINE or EOF."""
        tok = self._peek()
        if tok.kind not in (TokenKind.NEWLINE, TokenKind.EOF):
            self._error(
                f"unexpected extra input {self._describe(tok)} at end of line", tok
            )

    def _parse_statement(self):
        tok = self._peek()
        if tok.kind != TokenKind.IDENT:
            self._error(
                f"expected a gate or command at the start of a line, found "
                f"{self._describe(tok)}",
                tok,
            )

        word = tok.text
        if word == "qubits":
            self._parse_qubits_decl()
        elif word == "MEASURE":
            self._parse_measure()
        elif word in SOURCE_GATES:
            # Normalise aliases (e.g. CX -> CNOT) to a single canonical name so
            # every later stage only has to deal with one spelling.
            self._parse_gate(GATE_ALIASES.get(word, word))
        elif _QUBIT_RE.match(word):
            self._error(
                f"expected a gate or command at the start of a line, found qubit "
                f"reference '{word}'",
                tok,
            )
        else:
            self._error(f"unknown gate or command '{word}'", tok)

    # ---- individual statements -----------------------------------------------

    def _parse_qubits_decl(self):
        keyword = self._advance()  # consume 'qubits'
        if self.n_qubits is not None:
            self._error(
                "register size already declared; 'qubits' may appear only once",
                keyword,
            )

        num = self._peek()
        if num.kind != TokenKind.NUMBER:
            self._error(
                f"expected a register size after 'qubits', found {self._describe(num)}",
                num,
            )
        self._advance()

        value = float(num.text)
        if not value.is_integer() or value < 1:
            self._error(
                f"register size must be a positive whole number, found '{num.text}'",
                num,
            )
        self.n_qubits = int(value)

    def _parse_gate(self, name):
        keyword = self._advance()  # consume the gate name
        self._require_register_declared(keyword)

        n_qubits_needed, n_params_needed = GATE_SIGNATURES[name]

        params = ()
        if n_params_needed:
            params = (self._parse_paren_param(name, keyword),)

        qubits = self._parse_qubit_operands()

        if len(qubits) != n_qubits_needed:
            self._error(
                f"gate '{name}' expects {n_qubits_needed} "
                f"{_plural('qubit', n_qubits_needed)}, got {len(qubits)}",
                keyword,
            )

        if name == "CNOT" and qubits[0] == qubits[1]:
            self._error(
                f"CNOT control and target must be different qubits, both are q{qubits[0]}",
                keyword,
            )

        self.instructions.append(
            Gate(name, tuple(qubits), params, line=keyword.line, col=keyword.col)
        )

    def _parse_measure(self):
        keyword = self._advance()  # consume 'MEASURE'
        self._require_register_declared(keyword)

        qubits = self._parse_qubit_operands()
        if len(qubits) != 1:
            self._error(
                f"MEASURE expects 1 qubit, got {len(qubits)}", keyword
            )
        self.instructions.append(
            Measure(qubits[0], line=keyword.line, col=keyword.col)
        )

    # ---- shared operand parsing ----------------------------------------------

    def _parse_paren_param(self, name, keyword):
        """Parse the `(theta)` part of a parametrized gate like RZ."""
        lparen = self._peek()
        if lparen.kind != TokenKind.LPAREN:
            self._error(
                f"gate '{name}' takes a parameter; expected '(' after '{name}', "
                f"found {self._describe(lparen)}",
                lparen,
            )
        self._advance()

        num = self._peek()
        if num.kind != TokenKind.NUMBER:
            self._error(
                f"expected a numeric parameter inside '{name}(...)', found "
                f"{self._describe(num)}",
                num,
            )
        self._advance()
        value = float(num.text)

        rparen = self._peek()
        if rparen.kind != TokenKind.RPAREN:
            self._error(
                f"expected ')' to close '{name}(...)', found {self._describe(rparen)}",
                rparen,
            )
        self._advance()
        return value

    def _parse_qubit_operands(self):
        """Consume every remaining operand token on this line as a qubit ref.

        We greedily read until the NEWLINE/EOF so that *too many* operands are
        caught by the arity check in the caller (e.g. `H q0 q1`), rather than
        silently leaving a trailing token.
        """
        qubits = []
        while self._peek().kind not in (TokenKind.NEWLINE, TokenKind.EOF):
            qubits.append(self._parse_qubit_ref())
        return qubits

    def _parse_qubit_ref(self):
        tok = self._peek()
        if tok.kind != TokenKind.IDENT:
            self._error(
                f"expected a qubit like 'q0', found {self._describe(tok)}", tok
            )
        m = _QUBIT_RE.match(tok.text)
        if m is None:
            self._error(
                f"expected a qubit like 'q0', found '{tok.text}'", tok
            )
        self._advance()

        index = int(m.group(1))
        if index >= self.n_qubits:
            self._error(
                f"qubit index q{index} is out of range; register has "
                f"{self.n_qubits} {_plural('qubit', self.n_qubits)} "
                f"(q0..q{self.n_qubits - 1})",
                tok,
            )
        return index

    def _require_register_declared(self, tok):
        if self.n_qubits is None:
            self._error(
                "register size must be declared with 'qubits N' before any gates",
                tok,
            )

    # ---- misc -----------------------------------------------------------------

    @staticmethod
    def _describe(tok):
        """A short human description of a token for error messages."""
        if tok.kind == TokenKind.NEWLINE:
            return "end of line"
        if tok.kind == TokenKind.EOF:
            return "end of input"
        return f"'{tok.text}'"


def _plural(word, n):
    return word if n == 1 else word + "s"
