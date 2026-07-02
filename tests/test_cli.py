"""M7 CLI tests: drive `braket run` and check its output and exit codes."""

from pathlib import Path

import pytest

from braket.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run(capsys, argv):
    """Invoke the CLI, returning (exit_code, stdout, stderr)."""
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# --------------------------------------------------------------------------- #
# Happy paths on the shipped examples
# --------------------------------------------------------------------------- #

def test_run_bell_state(capsys):
    code, out, _ = run(capsys, ["run", str(EXAMPLES / "bell_state.bkt"),
                                "--shots", "1000", "--seed", "7"])
    assert code == 0
    # All pipeline stages are shown.
    for section in ["Parsed circuit", "Optimized circuit",
                    "Mapped circuit", "Native circuit",
                    "Final state vector", "Outcome probabilities",
                    "Sampled measurements"]:
        assert section in out
    # Native circuit really is native (H got rewritten into RZ/X90).
    assert "X90 q0" in out
    # Bell histogram: only correlated outcomes appear.
    assert "00 " in out and "11 " in out
    assert "01 " not in out and "10 " not in out
    # Self-check passed.
    assert "matches original statistics: OK" in out


def test_run_ghz_state(capsys):
    code, out, _ = run(capsys, ["run", str(EXAMPLES / "ghz_state.bkt"),
                                "--shots", "500", "--seed", "3"])
    assert code == 0
    assert "|000>: 0.5000" in out
    assert "|111>: 0.5000" in out
    assert "matches original statistics: OK" in out


def test_run_interference_is_deterministic(capsys):
    code, out, _ = run(capsys, ["run", str(EXAMPLES / "interference.bkt"),
                                "--shots", "200", "--seed", "1"])
    assert code == 0
    # Destructive interference on |0>, constructive on |1> -> always 1.
    assert "|1>: 1.0000" in out
    assert "1  " in out  # histogram row for outcome '1'


def test_no_shots_skips_histogram_but_shows_state(capsys):
    code, out, _ = run(capsys, ["run", str(EXAMPLES / "bell_state.bkt")])
    assert code == 0
    assert "Final state vector" in out
    assert "Sampled measurements" not in out


# --------------------------------------------------------------------------- #
# Stage-toggle flags
# --------------------------------------------------------------------------- #

def test_no_decompose_keeps_high_level_gates(capsys):
    code, out, _ = run(capsys, ["run", str(EXAMPLES / "bell_state.bkt"),
                                "--no-decompose"])
    assert code == 0
    assert "Native circuit" not in out
    # Without decomposition the H survives verbatim.
    assert "H q0" in out


def test_no_map_skips_routing_section(capsys):
    code, out, _ = run(capsys, ["run", str(EXAMPLES / "bell_state.bkt"),
                                "--no-map"])
    assert code == 0
    assert "Mapped circuit" not in out


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #

def test_missing_file_returns_error(capsys):
    code, _, err = run(capsys, ["run", str(EXAMPLES / "nope.bkt")])
    assert code == 1
    assert "cannot read" in err


def test_parse_error_is_reported_with_location(capsys, tmp_path):
    bad = tmp_path / "bad.bkt"
    bad.write_text("qubits 2\nFOO q0\n")
    code, _, err = run(capsys, ["run", str(bad)])
    assert code == 1
    assert "parse error" in err
    assert "line 2" in err
    assert "FOO" in err


def test_out_of_range_qubit_is_reported(capsys, tmp_path):
    bad = tmp_path / "oor.bkt"
    bad.write_text("qubits 2\nH q5\n")
    code, _, err = run(capsys, ["run", str(bad)])
    assert code == 1
    assert "out of range" in err
