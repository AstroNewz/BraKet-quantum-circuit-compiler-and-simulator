"""Lexer: source text -> list of tokens (M1).

Reads a `.bkt` circuit character by character and groups them into `Token`s,
tracking line and column so later stages can report *where* a problem is.
Comments (`# ... to end of line`) and insignificant whitespace are dropped here.

The implementation processes the source one line at a time. That makes column
tracking trivial (the column is just the offset within the current line) and
lets us emit a NEWLINE token between lines so the parser can tell statements
apart.
"""

import re

from .errors import LexError
from .tokens import Token, TokenKind

# One master regex whose named groups map onto token kinds. `re.match` tries the
# alternatives left to right at the current position, so ordering matters:
#   - NUMBER must come before IDENT is irrelevant (they start with different
#     characters), but we still list the "structural" pieces explicitly.
#   - WS and COMMENT are matched so we can *skip* them; they produce no token.
#
# NUMBER accepts optional sign, integer/decimal/leading-dot forms, and an
# optional exponent — enough for angle literals like 1.5708, -3.14, .5, 1e-3.
_TOKEN_RE = re.compile(
    r"""
      (?P<WS>[ \t\r]+)
    | (?P<COMMENT>\#[^\n]*)
    | (?P<NUMBER>[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
    | (?P<LPAREN>\()
    | (?P<RPAREN>\))
    | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)

# Regex group name -> token kind, for the groups that actually become tokens.
_KIND_BY_GROUP = {
    "NUMBER": TokenKind.NUMBER,
    "LPAREN": TokenKind.LPAREN,
    "RPAREN": TokenKind.RPAREN,
    "IDENT": TokenKind.IDENT,
}


def tokenize(source):
    """Turn circuit source text into a list of tokens ending with EOF.

    Raises `LexError` (with line/col and the offending source line) on any
    character that cannot start a token.
    """
    tokens = []
    lines = source.split("\n")

    for line_index, line_text in enumerate(lines):
        lineno = line_index + 1  # 1-based line number
        pos = 0
        while pos < len(line_text):
            match = _TOKEN_RE.match(line_text, pos)
            if match is None:
                # Nothing in our alphabet starts here -> an illegal character.
                bad_char = line_text[pos]
                raise LexError(
                    f"unexpected character {bad_char!r}",
                    line=lineno,
                    col=pos + 1,
                    source_line=line_text,
                )

            group = match.lastgroup
            if group not in ("WS", "COMMENT"):
                tokens.append(
                    Token(
                        kind=_KIND_BY_GROUP[group],
                        text=match.group(),
                        line=lineno,
                        col=pos + 1,
                    )
                )
            pos = match.end()

        # Mark the end of every source line. The parser treats a run of NEWLINEs
        # (blank/comment-only lines) as a single separator, so this is harmless
        # even for empty lines.
        tokens.append(
            Token(TokenKind.NEWLINE, "", line=lineno, col=len(line_text) + 1)
        )

    tokens.append(Token(TokenKind.EOF, "", line=len(lines), col=1))
    return tokens
