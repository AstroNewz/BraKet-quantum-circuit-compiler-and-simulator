"""BraKet — a small quantum circuit compiler and state-vector simulator.

The public pipeline is built up milestone by milestone. Each stage lives in its
own module so it can be read, tested, and explained independently:

    lexer      -> tokenize source text
    parser     -> tokens into a Circuit AST
    optimizer  -> simplify the circuit (cancel/merge gates)
    mapper     -> route 2-qubit gates onto a hardware topology
    decomposer -> rewrite into a native gate set
    simulator  -> evolve the quantum state and sample measurements
"""

__version__ = "0.1.0"
