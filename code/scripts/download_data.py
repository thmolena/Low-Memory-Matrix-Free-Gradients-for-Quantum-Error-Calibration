#!/usr/bin/env python3
"""Fetch and authenticate the four registered SuiteSparse PARSEC matrices."""
from __future__ import annotations

from rc_krylov_qec.hamiltonians import PARSEC_MATRICES, download_matrix


def main() -> None:
    for name in PARSEC_MATRICES:
        matrix, digest = download_matrix(name, verify=True)
        print(
            f"{name}: n={matrix.shape[0]}, nnz={matrix.nnz}, "
            f"sha256={digest}"
        )


if __name__ == "__main__":
    main()
