"""M0 smoke tests: the scaffold imports cleanly and the toolchain is wired up.

These are deliberately trivial. Their only job is to prove that the package
layout is importable and pytest runs green before we start adding real logic in
M1. Each later milestone replaces "todo" checks with real behavioural tests.
"""

import numpy as np

import braket
from braket import (
    ast_nodes,
    cli,
    decomposer,
    errors,
    gates,
    lexer,
    mapper,
    optimizer,
    parser,
    simulator,
    tokens,
)


def test_version_is_exposed():
    assert braket.__version__ == "0.1.0"


def test_all_pipeline_modules_import():
    # Every stage of the pipeline should at least be importable from day one.
    for module in (
        tokens,
        lexer,
        ast_nodes,
        parser,
        gates,
        simulator,
        optimizer,
        mapper,
        decomposer,
        cli,
        errors,
    ):
        assert module is not None


def test_numpy_is_available():
    # numpy underpins the M2 simulator; fail loudly here if the env is wrong.
    assert np.allclose(np.eye(2) @ np.eye(2), np.eye(2))


def test_base_error_type_exists():
    assert issubclass(errors.BraKetError, Exception)
