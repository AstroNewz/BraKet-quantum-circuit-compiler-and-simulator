"""M1 lexer tests.

We check that the lexer produces the right token *kinds* and *values*, drops
comments and whitespace, tracks line/column, and rejects illegal characters with
a located error.
"""

import pytest

from braket.errors import LexError
from braket.lexer import tokenize
from braket.tokens import TokenKind


def kinds(tokens):
    return [t.kind for t in tokens]


def test_simple_line_tokens():
    toks = tokenize("H q0")
    assert kinds(toks) == [
        TokenKind.IDENT,   # H
        TokenKind.IDENT,   # q0
        TokenKind.NEWLINE,
        TokenKind.EOF,
    ]
    assert toks[0].text == "H"
    assert toks[1].text == "q0"


def test_rz_produces_number_and_parens():
    toks = tokenize("RZ(1.5708) q0")
    assert kinds(toks)[:5] == [
        TokenKind.IDENT,    # RZ
        TokenKind.LPAREN,
        TokenKind.NUMBER,   # 1.5708
        TokenKind.RPAREN,
        TokenKind.IDENT,    # q0
    ]
    assert toks[2].text == "1.5708"


def test_negative_and_dotted_and_exponent_numbers():
    for literal in ["-3.14", ".5", "1e-3", "+2", "0"]:
        toks = tokenize(f"RZ({literal}) q0")
        number = toks[2]
        assert number.kind == TokenKind.NUMBER
        assert number.text == literal


def test_comments_and_blank_lines_are_dropped():
    source = "# a comment\nqubits 2   # trailing comment\n\nH q0\n"
    texts = [t.text for t in tokenize(source) if t.kind == TokenKind.IDENT]
    assert texts == ["qubits", "H", "q0"]


def test_line_and_column_tracking():
    # Second line, the 'H' starts at column 1; 'q0' after "H " at column 3.
    toks = tokenize("qubits 2\nH q0")
    h_tok = [t for t in toks if t.text == "H"][0]
    q_tok = [t for t in toks if t.text == "q0"][0]
    assert (h_tok.line, h_tok.col) == (2, 1)
    assert (q_tok.line, q_tok.col) == (2, 3)


def test_newline_between_lines_and_final_eof():
    toks = tokenize("qubits 1\nX q0")
    assert toks[-1].kind == TokenKind.EOF
    assert any(t.kind == TokenKind.NEWLINE for t in toks)


def test_illegal_character_raises_located_lexerror():
    with pytest.raises(LexError) as exc:
        tokenize("qubits 2\nH @q0")
    err = exc.value
    assert err.line == 2
    assert err.col == 3          # the '@'
    assert "@" in err.message
    # str() renders a caret pointing at the bad character.
    assert "^" in str(err)
