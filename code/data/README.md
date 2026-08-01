# Data provenance

**Dataset:** SuiteSparse PARSEC matrices Si2, SiH4, benzene, and Si5H12.

**Original source:** https://sparse.tamu.edu/PARSEC

**Study design:** The cross-matrix experiment compares the direct derivative of
the finite Lanczos objective with an independently rerun centered difference
and with a reused-basis forward derivative at six Krylov depths.  Separate
experiments test the unprojected error bound on Si2, record the large-object
storage ledger on all four matrices, and alternate dense and matrix-free timing
runs on Si2.

**Integrity:** The real-data command validates every archive against a pinned digest.

Run from the repository root:

```bash
python -m pip install -e code
python code/scripts/download_data.py
```

Downloaded third-party files remain governed by the source terms documented in
`THIRD_PARTY.md`. When redistribution is not explicit, the package fetches
the data into an external cache instead of committing the source bytes.
