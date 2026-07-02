"""Token definitions for the BraKet lexer (M1).

A *token* is the smallest meaningful chunk of source text — a word like `qubits`
or `H`, a number, a parenthesis, or a line break. The lexer turns raw characters
into a flat list of these; the parser then reads the list to build the AST.

Each token remembers its 1-based `line` and `col` so that any later error can
point back at exactly where the token came from in the source.
"""

from dataclasses import dataclass
from enum import Enum


class TokenKind(Enum):
    """The categories of token BraKet's lexer produces.

    We deliberately keep this tiny. Notably there is no separate "keyword" or
    "gate" kind: words like `qubits`, `H`, and the qubit reference `q0` are all
    `IDENT`s. Deciding what a given word *means* is the parser's job, which keeps
    the lexer dumb and easy to reason about.
    """

    IDENT = "IDENT"      # a word: qubits, H, X, RZ, MEASURE, q0, ...
    NUMBER = "NUMBER"    # a numeric literal: 2, 1.5708, -3.14, .5, 1e-3
    LPAREN = "LPAREN"    # (
    RPAREN = "RPAREN"    # )
    NEWLINE = "NEWLINE"  # end of a source line (statement separator)
    EOF = "EOF"          # end of input


@dataclass(frozen=True)
class Token:
    """One lexical token plus where it appeared in the source.

    `text` is the exact substring matched (e.g. "1.5708" or "CNOT"). `line` and
    `col` are 1-based. NEWLINE/EOF tokens carry an empty `text`.
    """

    kind: TokenKind
    text: str
    line: int
    col: int

    def __repr__(self):
        return f"Token({self.kind.name}, {self.text!r}, line={self.line}, col={self.col})"
