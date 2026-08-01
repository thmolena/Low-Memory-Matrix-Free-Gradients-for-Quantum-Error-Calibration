# Parameter-Blocked Lanczos Sensitivity Replay

This repository contains the manuscript and reproducibility package for parameter-blocked differentiation of a finite Lanczos matrix-function objective.

The method propagates up to `q` parameter tangents as columns of one multivector. For `p` parameters and `m` Lanczos steps, the Hamiltonian launch count is

\[
m + 2m\lceil p/q\rceil.
\]

Parameterwise replay is the `q=1` endpoint. Full blocking is the `q=p` endpoint and uses `3m` Hamiltonian launches independent of `p`. The working-memory cost grows from `O(n)` to `O(qn)`, so block width exposes an explicit launch–storage frontier.

## Locked numerical result

- Four checksum-pinned PARSEC matrices: Si2, SiH4, benzene, and Si5H12.
- Four Lanczos depths: 8, 16, 32, and 48.
- Maximum full-block versus scalar gradient difference: `2.50e-16`.
- Scalar launch count at `p=3`: `7m`.
- Full-block launch count at `p=3`: `3m`.
- Median CPU speedups at `p=3`, `m=32`: `1.10x`, `1.21x`, `1.25x`, and `1.06x`.
- At `p=8`, `m=24` on SiH4: 408 launches reduce to 72, and median time reduces from 0.224 s to 0.103 s.
- Launch reduction is exact; wall time remains host and kernel dependent.

Semantic result hash:

```text
3dd1bab58150136019fd61ac6ceed799f152dabdc194d68ce61145dc85770213
```

## Installation and replay

```bash
cd code
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,reproduce]'
python -m pytest -q
cd ..
rc-krylov-block-study --verify
```

The downloader reuses `~/.cache/rc-krylov-qec/suitesparse` by default and verifies every PARSEC archive against its registered SHA-256 digest.

## Artifact map

- `main.tex`, `main.pdf`: blank-page manuscript rewrite and compiled paper.
- `code/src/rc_krylov_qec/block_replay.py`: parameter-blocked tangent replay.
- `code/src/rc_krylov_qec/block_study.py`: authenticated matrix study and figure generation.
- `code/tests/test_block_replay.py`: exact-equivalence and operator-ledger tests.
- `code/results/block_replay/block_replay_study.json`: locked accuracy, launch, storage, timing, and environment record.
- `code/manuscript_assets/figures/block_equivalence.*`: finite-gradient equivalence.
- `code/manuscript_assets/figures/block_runtime.*`: real-matrix timing.
- `code/manuscript_assets/figures/launch_memory_tradeoff.*`: proved launch–storage frontier.
- `code/manuscript_assets/figures/parameter_runtime.*`: measured parameter scaling.

## Scientific scope

The PARSEC matrices are authentic sparse electronic-structure operators. The diagonal parameter directions and probe define a transparent semi-synthetic calibration objective. They are not device measurements. Timing is host dependent and is retained separately from the semantic equality contract.

The source is released under the MIT License. External matrices remain governed by the SuiteSparse Matrix Collection terms.
