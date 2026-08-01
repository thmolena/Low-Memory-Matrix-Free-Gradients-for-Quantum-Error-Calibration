# Reproducibility package

The primary executable is `rc-krylov-block-study`. It authenticates the PARSEC archives, compares parameter-blocked and parameterwise replay, writes the result ledger, and regenerates the four manuscript figures.

```bash
python -m pip install -e '.[test,reproduce]'
python -m pytest -q
cd ..
rc-krylov-block-study --verify
```

The semantic result hash is `3dd1bab58150136019fd61ac6ceed799f152dabdc194d68ce61145dc85770213`. Timing values are recorded but excluded from semantic equality.

The earlier `rc-krylov-qec-reproduce` entry point remains available for the scalar direct-replay and Chebyshev–Bessel projection-bound study.
