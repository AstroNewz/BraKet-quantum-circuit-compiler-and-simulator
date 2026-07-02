"""Error types for BraKet.

We use a small hierarchy so each pipeline stage can raise something specific,
while callers (the CLI, tests) can catch the common base class `BraKetError`.

The interesting part is that errors carry a source *location* (line + column) and,
when available, the offending source line. `str(error)` then renders the kind of
message a real compiler gives — pointing a caret at exactly where things went
wrong. For example:

    line 3, col 1: unknown gate 'FOO'
        FOO q0
        ^

Keeping the raw message on `.message` (separate from the rendered string) makes
it easy for tests to assert on the human-readable text without caring about the
caret formatting.
"""


class BraKetError(Exception):
    """Base class for every error BraKet raises on purpose.

    Parameters
    ----------
    message:
        The human-readable description of what went wrong (no location prefix).
    line, col:
        1-based source coordinates, if known. `line` alone is allowed (points at
        a whole line); a caret is only drawn when `col` is also present.
    source_line:
        The full text of the offending source line, used to draw the caret. If
        omitted we still print the message and coordinates, just without the
        visual pointer.
    """

    def __init__(self, message, line=None, col=None, source_line=None):
        self.message = message
        self.line = line
        self.col = col
        self.source_line = source_line
        super().__init__(self._render())

    def _render(self):
        # No location info: just the plain message.
        if self.line is None:
            return self.message

        location = f"line {self.line}"
        if self.col is not None:
            location += f", col {self.col}"
        rendered = f"{location}: {self.message}"

        # Draw a caret under the offending column when we have both the source
        # text and a column. `col` is 1-based, so col-1 spaces precede the caret.
        if self.source_line is not None and self.col is not None:
            caret_pad = " " * (self.col - 1)
            rendered += f"\n    {self.source_line}\n    {caret_pad}^"
        return rendered


class LexError(BraKetError):
    """The lexer hit a character it could not turn into a token."""


class ParseError(BraKetError):
    """The parser found tokens that do not form a valid circuit.

    Covers both grammar problems (wrong shape) and the static semantic checks we
    can do without simulating (qubit index out of range, wrong argument count,
    duplicate register declaration, ...).
    """
