"""Real Hermitian Hamiltonians from the SuiteSparse Matrix Collection.

The matrices are real-space pseudopotential density-functional Hamiltonians
produced by the PARSEC electronic-structure code and distributed, with a
permanent identifier and checksum, by the SuiteSparse Matrix Collection
(Davis and Hu).  They are large, sparse, and exactly symmetric, which makes
them authentic stress tests for the matrix-free gradient primitive with an error bound:
the estimator only ever multiplies by ``H`` and by the derivative operators,
so no dense matrix of the ambient dimension is ever formed.

Everything here runs on a laptop CPU.  A matrix is downloaded once from the
public collection, checked against a pinned SHA-256 digest, cached, and then
loaded straight from the Matrix Market file inside the archive.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.io import mmread
from scipy.sparse.linalg import eigsh

ComplexVector = NDArray[np.complex128]
Matvec = Callable[[ComplexVector], ComplexVector]

#: Base URL of the SuiteSparse Matrix Collection Matrix Market archives.
SUITESPARSE_BASE_URL = (
    "https://suitesparse-collection-website.herokuapp.com/MM"
)


@dataclass(frozen=True)
class MatrixSpec:
    """Identifier and pinned checksum for one collection matrix."""

    group: str
    name: str
    dimension: int
    sha256: str

    @property
    def identifier(self) -> str:
        return f"{self.group}/{self.name}"

    @property
    def url(self) -> str:
        return f"{SUITESPARSE_BASE_URL}/{self.group}/{self.name}.tar.gz"


#: PARSEC real-symmetric DFT Hamiltonians used in the real-data study.  The
#: digests pin the exact public archives so a reproduction fails loudly if the
#: upstream file ever changes.
PARSEC_MATRICES: dict[str, MatrixSpec] = {
    "Si2": MatrixSpec(
        "PARSEC", "Si2", 769,
        "84e9c6f4b61a88adc1fc927e3be1fa75cb51d01f4ba26287e9c803578c0e27cd",
    ),
    "SiH4": MatrixSpec(
        "PARSEC", "SiH4", 5041,
        "803b52431d5e4967ac230cd5be591557d7a99c175b9ec170555151f8d76776bd",
    ),
    "benzene": MatrixSpec(
        "PARSEC", "benzene", 8219,
        "a1f78f39eefc4436820d9a9745e2b7c73375dce54e6bf54c6d39da0b029e89f4",
    ),
    "Si5H12": MatrixSpec(
        "PARSEC", "Si5H12", 19896,
        "895c0d8c7069e4a0401fc68838a124fbec7e4bc77f2dbd581cea8828e47578b9",
    ),
}


def default_cache_dir() -> Path:
    """Return the cache directory for downloaded matrices."""

    override = os.environ.get("RC_KRYLOV_QEC_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "rc-krylov-qec" / "suitesparse"


def download_matrix(
    name: str,
    *,
    cache_dir: Path | None = None,
    verify: bool = True,
) -> tuple[sp.csr_matrix, str]:
    """Fetch, checksum, cache, and load a PARSEC matrix as a CSR array.

    Returns the exactly symmetric sparse matrix and the observed SHA-256
    digest of the downloaded archive.
    """

    if name not in PARSEC_MATRICES:
        raise KeyError(
            f"unknown matrix {name!r}; available: "
            f"{sorted(PARSEC_MATRICES)}"
        )
    spec = PARSEC_MATRICES[name]
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"{spec.group}_{spec.name}.tar.gz"

    if not archive.exists():
        urllib.request.urlretrieve(spec.url, archive)  # noqa: S310 (https)

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if verify and digest != spec.sha256:
        raise RuntimeError(
            f"checksum mismatch for {spec.identifier}: expected "
            f"{spec.sha256}, downloaded {digest}. Delete {archive} and retry."
        )

    with tarfile.open(archive, "r:gz") as handle:
        members = [m for m in handle.getmembers() if m.name.endswith(".mtx")]
        if not members:
            raise RuntimeError(f"no Matrix Market file inside {archive}")
        stream = handle.extractfile(members[0])
        matrix = mmread(stream).tocsr().astype(np.float64)

    # Matrix Market symmetric storage already yields a symmetric matrix; the
    # explicit symmetrization guards against round-off in the reader.
    matrix = (0.5 * (matrix + matrix.T)).tocsr()
    if matrix.shape[0] != spec.dimension:
        raise RuntimeError(
            f"unexpected dimension {matrix.shape[0]} for {spec.identifier}"
        )
    return matrix, digest


def _extremal_eigenvalues(matrix: sp.csr_matrix) -> tuple[float, float]:
    """Return deterministic estimates of the smallest and largest eigenvalues."""

    # A fixed starting vector makes the Lanczos-based extremal solves
    # reproducible across runs.
    start = np.ones(matrix.shape[0], dtype=np.float64)
    start /= np.linalg.norm(start)
    largest = float(
        eigsh(
            matrix, k=1, which="LA", v0=start, maxiter=5000, tol=0,
            return_eigenvectors=False,
        )[0]
    )
    smallest = float(
        eigsh(
            matrix, k=1, which="SA", v0=start, maxiter=10000, tol=0,
            return_eigenvectors=False,
        )[0]
    )
    return smallest, largest


@dataclass(frozen=True)
class RealCalibrationProblem:
    """A parameterized survival objective built on a real Hamiltonian.

    The Hamiltonian is ``H(theta) = H0 + sum_a theta_a P_a`` where ``H0`` is the
    spectrally rescaled PARSEC matrix and each ``P_a`` is a diagonal projector
    onto a contiguous block of the index set.  A projector has spectral norm
    exactly one, so the compression certificate bound is the bare nuclear tail.
    Physically the parameters are localized potential offsets, the coherent
    analogue of a per-region miscalibration.
    """

    name: str
    dimension: int
    matvec: Matvec
    derivative_matvecs: tuple[Matvec, ...]
    probe: ComplexVector
    theta_true: NDArray[np.float64]
    operator_norms: NDArray[np.float64]
    hamiltonian_norm_upper_bound: float
    time: float
    center: float
    scale: float
    lambda_min: float
    lambda_max: float
    sha256: str

    @property
    def n_parameters(self) -> int:
        return self.theta_true.size


def _region_projectors(
    dimension: int, n_regions: int
) -> list[NDArray[np.float64]]:
    """Return indicator diagonals of a disjoint contiguous partition."""

    blocks = np.array_split(np.arange(dimension), n_regions)
    diagonals = []
    for block in blocks:
        diagonal = np.zeros(dimension, dtype=np.float64)
        diagonal[block] = 1.0
        diagonals.append(diagonal)
    return diagonals


def assemble_problem(
    name: str,
    matrix: sp.spmatrix,
    *,
    sha256: str = "",
    time: float = 8.0,
    theta_true: Sequence[float] = (0.05, -0.03, 0.02),
    probe_seed: int = 20260726,
) -> RealCalibrationProblem:
    """Build a matrix-free survival objective from an in-memory matrix.

    Factored out of :func:`build_real_problem` so the study can be exercised on
    a small synthetic symmetric matrix without any network access.
    """

    matrix = matrix.tocsr().astype(np.float64)
    dimension = matrix.shape[0]
    lambda_min, lambda_max = _extremal_eigenvalues(matrix)
    center = 0.5 * (lambda_max + lambda_min)
    scale = 0.5 * (lambda_max - lambda_min)
    if scale <= 0.0:
        raise RuntimeError("degenerate spectrum; cannot rescale")

    identity = sp.eye(dimension, format="csr")
    scaled = ((matrix - center * identity) / scale).tocsr()

    truth = np.asarray(theta_true, dtype=np.float64)
    diagonals = _region_projectors(dimension, truth.size)
    projectors = [sp.diags(diagonal, format="csr") for diagonal in diagonals]
    # Symmetry gives ||scaled||_2 <= ||scaled||_infinity.  Because the
    # projectors form a disjoint diagonal partition, the perturbation norm is
    # max_a |theta_a|.  Round one float upward before exposing the certificate
    # input.
    computed_infinity_norm = float(
        np.max(np.asarray(abs(scaled).sum(axis=1)).ravel())
    )
    maximum_row_terms = int(np.max(np.diff(scaled.indptr)))
    summation_gamma = (
        maximum_row_terms * np.finfo(float).eps
        / (1.0 - maximum_row_terms * np.finfo(float).eps)
    )
    scaled_infinity_norm = float(
        np.nextafter(
            computed_infinity_norm / (1.0 - summation_gamma), np.inf
        )
    )
    hamiltonian_norm_upper_bound = float(
        np.nextafter(
            scaled_infinity_norm + float(np.max(np.abs(truth))), np.inf
        )
    )

    def matvec(vector: ComplexVector) -> ComplexVector:
        vector = np.asarray(vector, dtype=np.complex128)
        result = scaled @ vector
        for coefficient, projector in zip(truth, projectors):
            result = result + coefficient * (projector @ vector)
        return result.astype(np.complex128)

    def make_derivative(projector: sp.csr_matrix) -> Matvec:
        def derivative(vector: ComplexVector) -> ComplexVector:
            return (projector @ np.asarray(vector, dtype=np.complex128)).astype(
                np.complex128
            )

        return derivative

    derivative_matvecs = tuple(make_derivative(p) for p in projectors)

    rng = np.random.default_rng(probe_seed)
    probe = rng.standard_normal(dimension).astype(np.complex128)
    probe /= np.linalg.norm(probe)

    return RealCalibrationProblem(
        name=name,
        dimension=dimension,
        matvec=matvec,
        derivative_matvecs=derivative_matvecs,
        probe=probe,
        theta_true=truth,
        operator_norms=np.ones(truth.size, dtype=np.float64),
        hamiltonian_norm_upper_bound=hamiltonian_norm_upper_bound,
        time=float(time),
        center=center,
        scale=scale,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        sha256=sha256,
    )


def build_real_problem(
    name: str,
    *,
    time: float = 8.0,
    theta_true: Sequence[float] = (0.05, -0.03, 0.02),
    probe_seed: int = 20260726,
    cache_dir: Path | None = None,
    verify: bool = True,
) -> RealCalibrationProblem:
    """Download a PARSEC matrix and assemble its survival objective."""

    matrix, digest = download_matrix(name, cache_dir=cache_dir, verify=verify)
    return assemble_problem(
        name,
        matrix,
        sha256=digest,
        time=time,
        theta_true=theta_true,
        probe_seed=probe_seed,
    )


def dense_reference_hamiltonian(
    problem: RealCalibrationProblem,
) -> NDArray[np.complex128]:
    """Materialize ``H(theta_true)`` densely.

    Used only for the small-matrix finite-difference oracle; never called by
    the estimator itself.
    """

    dimension = problem.dimension
    basis = np.eye(dimension, dtype=np.complex128)
    columns = [problem.matvec(basis[:, index]) for index in range(dimension)]
    return np.column_stack(columns)
