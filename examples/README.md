# Example circuits

Run any of these through the full compiler + simulator pipeline:

```bash
braket run examples/bell_state.bkt --shots 1000 --seed 7
# or without installing the entry point:
python -m braket.cli run examples/bell_state.bkt --shots 1000 --seed 7
```

| File                | What it shows                                                        |
|---------------------|---------------------------------------------------------------------|
| `bell_state.bkt`    | Two-qubit entanglement: outcomes are always `00` or `11` (50/50).   |
| `ghz_state.bkt`     | Three-qubit entanglement: always `000` or `111` (50/50).            |
| `interference.bkt`  | Single-qubit interference: a relative phase flips the result to `1`.|

Useful flags: `--shots N` (sample N measurements), `--seed S` (reproducible
sampling), and `--no-optimize` / `--no-map` / `--no-decompose` to watch the
circuit with individual compiler stages turned off.
