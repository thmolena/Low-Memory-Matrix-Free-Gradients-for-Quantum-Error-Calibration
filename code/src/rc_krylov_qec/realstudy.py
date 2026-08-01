"""Authenticated PARSEC study for direct tridiagonal-sensitivity replay.

The study tests four predeclared hypotheses on four real sparse Hamiltonians
from the SuiteSparse Matrix Collection.  The direct replay is compared with
the reused-basis forward derivative and an independently centered finite
difference of the finite Lanczos objective.  On the smallest matrix, a dense
Frechet derivative supplies an additional unprojected reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.linalg import expm, expm_frechet

from .hamiltonians import (
    PARSEC_MATRICES,
    RealCalibrationProblem,
    build_real_problem,
    dense_reference_hamiltonian,
    download_matrix,
)
from .low_memory import (
    direct_tridiagonal_replay_gradient,
    error_bounded_survival_gradient,
    low_memory_lanczos,
)
from .projected import survival_probability_sensitivity

DEPTHS = (6, 8, 10, 12, 14, 16)
BOUND_DEPTHS = (6, 8, 10, 12, 14, 16, 18, 20)
FINITE_DIFFERENCE_STEP = 2.0e-6
STORAGE_STEPS = 48
TIMING_STEPS = 14
TIMING_REPEATS = 7
COMPLEX_BYTES = 16


def _dense_arrays(
    problem: RealCalibrationProblem,
) -> tuple[NDArray[np.complex128], list[NDArray[np.complex128]]]:
    hamiltonian = dense_reference_hamiltonian(problem)
    basis = np.eye(problem.dimension, dtype=np.complex128)
    directions = [
        np.column_stack([action(basis[:, j]) for j in range(problem.dimension)])
        for action in problem.derivative_matvecs
    ]
    return hamiltonian, directions


def _dense_frechet_from_arrays(
    problem: RealCalibrationProblem,
    hamiltonian: NDArray[np.complex128],
    directions: list[NDArray[np.complex128]],
) -> tuple[float, NDArray[np.float64]]:
    generator = -1j * problem.time * hamiltonian
    propagator = expm(generator)
    amplitude = np.vdot(problem.probe, propagator @ problem.probe)
    gradient = np.empty(problem.n_parameters, dtype=np.float64)
    for index, direction in enumerate(directions):
        tangent = expm_frechet(
            generator,
            -1j * problem.time * direction,
            compute_expm=False,
        )
        amplitude_tangent = np.vdot(problem.probe, tangent @ problem.probe)
        gradient[index] = 2.0 * np.real(
            np.conj(amplitude) * amplitude_tangent
        )
    if not np.isfinite(amplitude) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("dense Frechet reference is nonfinite")
    return float(abs(amplitude) ** 2), gradient


def dense_frechet_gradient(
    problem: RealCalibrationProblem,
) -> tuple[float, NDArray[np.float64]]:
    """Independent dense reference for validation-sized matrices."""

    hamiltonian, directions = _dense_arrays(problem)
    return _dense_frechet_from_arrays(problem, hamiltonian, directions)


def _projected_probability(
    problem: RealCalibrationProblem,
    steps: int,
    parameter: int,
    shift: float,
) -> float:
    direction = problem.derivative_matvecs[parameter]

    def shifted(vector: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return problem.matvec(vector) + shift * direction(vector)

    coefficients = low_memory_lanczos(shifted, problem.probe, steps)
    probability, _ = survival_probability_sensitivity(
        coefficients.start_norm,
        coefficients.tridiagonal,
        problem.time,
    )
    return probability


def finite_difference_projected_gradient(
    problem: RealCalibrationProblem,
    steps: int,
    *,
    difference_step: float = FINITE_DIFFERENCE_STEP,
) -> NDArray[np.float64]:
    """Centered finite difference of the independently rerun Lanczos map."""

    gradient = np.empty(problem.n_parameters, dtype=np.float64)
    for parameter in range(problem.n_parameters):
        plus = _projected_probability(
            problem, steps, parameter, difference_step
        )
        minus = _projected_probability(
            problem, steps, parameter, -difference_step
        )
        gradient[parameter] = (plus - minus) / (2.0 * difference_step)
    return gradient


def run_cross_matrix_study(
    problems: dict[str, RealCalibrationProblem],
) -> list[dict[str, float | str]]:
    """Test the differentiated-recurrence identity on every real matrix."""

    rows: list[dict[str, float | str]] = []
    for name, problem in problems.items():
        for steps in DEPTHS:
            direct = direct_tridiagonal_replay_gradient(
                problem.matvec,
                problem.probe,
                problem.derivative_matvecs,
                problem.time,
                steps,
            )
            forward = error_bounded_survival_gradient(
                problem.matvec,
                problem.probe,
                problem.derivative_matvecs,
                problem.time,
                steps,
                steps,
                include_residual_correction=False,
            )
            reference = finite_difference_projected_gradient(problem, steps)
            direct_error = float(np.linalg.norm(direct.gradient - reference))
            forward_error = float(np.linalg.norm(forward.gradient - reference))
            rows.append(
                {
                    "matrix": name,
                    "dimension": float(problem.dimension),
                    "steps": float(steps),
                    "finite_difference_step": FINITE_DIFFERENCE_STEP,
                    "reference_gradient_norm": float(np.linalg.norm(reference)),
                    "direct_replay_error": direct_error,
                    "forward_only_error": forward_error,
                    "forward_over_direct_error": float(
                        forward_error / max(direct_error, np.finfo(float).tiny)
                    ),
                    "direct_hamiltonian_matvecs": float(
                        direct.hamiltonian_matvecs
                    ),
                    "direct_derivative_matvecs": float(
                        direct.derivative_matvecs
                    ),
                }
            )
    return rows


def run_bound_study(
    problem: RealCalibrationProblem,
    dense_gradient: NDArray[np.float64],
) -> list[dict[str, float | str]]:
    """Test the unprojected exact-arithmetic bound across depth."""

    rows: list[dict[str, float | str]] = []
    for steps in BOUND_DEPTHS:
        direct = direct_tridiagonal_replay_gradient(
            problem.matvec,
            problem.probe,
            problem.derivative_matvecs,
            problem.time,
            steps,
            derivative_operator_norms=problem.operator_norms,
            hamiltonian_norm_upper_bound=(
                problem.hamiltonian_norm_upper_bound
            ),
        )
        if (
            direct.krylov_component_error_bound is None
            or direct.krylov_vector_error_bound is None
        ):
            raise RuntimeError("Krylov bound was not computed")
        component_error = np.abs(direct.gradient - dense_gradient)
        rows.append(
            {
                "matrix": problem.name,
                "dimension": float(problem.dimension),
                "steps": float(steps),
                "observed_gradient_l2_error": float(
                    np.linalg.norm(component_error)
                ),
                "gradient_error_bound": float(
                    direct.krylov_vector_error_bound
                ),
                "maximum_component_error": float(np.max(component_error)),
                "minimum_component_slack": float(
                    np.min(
                        direct.krylov_component_error_bound - component_error
                    )
                ),
                "bound_covered": float(
                    np.all(
                        component_error
                        <= direct.krylov_component_error_bound
                        + 64.0 * np.finfo(float).eps
                    )
                ),
                "hamiltonian_norm_upper_bound": float(
                    problem.hamiltonian_norm_upper_bound
                ),
            }
        )
    return rows


def run_storage_study(
    problems: dict[str, RealCalibrationProblem],
) -> list[dict[str, float | str]]:
    """Execute the direct replay and record its allocation ledger."""

    rows: list[dict[str, float | str]] = []
    for name, problem in problems.items():
        started = perf_counter()
        direct = direct_tridiagonal_replay_gradient(
            problem.matvec,
            problem.probe,
            problem.derivative_matvecs,
            problem.time,
            STORAGE_STEPS,
        )
        elapsed = perf_counter() - started
        full_vectors = float(STORAGE_STEPS)
        direct_vectors = float(direct.estimated_peak_vector_equivalents)
        rows.append(
            {
                "matrix": name,
                "dimension": float(problem.dimension),
                "steps": float(STORAGE_STEPS),
                "full_basis_vector_equivalents": full_vectors,
                "direct_replay_vector_equivalents": direct_vectors,
                "full_basis_mib": float(
                    full_vectors * problem.dimension * COMPLEX_BYTES / 2**20
                ),
                "direct_replay_mib": float(
                    direct_vectors * problem.dimension * COMPLEX_BYTES / 2**20
                ),
                "storage_reduction": float(full_vectors / direct_vectors),
                "elapsed_seconds": float(elapsed),
                "hamiltonian_matvecs": float(direct.hamiltonian_matvecs),
                "derivative_matvecs": float(direct.derivative_matvecs),
                "sha256": problem.sha256,
            }
        )
    return rows


def run_timing_study(
    problem: RealCalibrationProblem,
    hamiltonian: NDArray[np.complex128],
    directions: list[NDArray[np.complex128]],
) -> list[dict[str, float | str]]:
    """Alternate direct replay and a preassembled dense Frechet reference."""

    _dense_frechet_from_arrays(problem, hamiltonian, directions)
    direct = direct_tridiagonal_replay_gradient(
        problem.matvec,
        problem.probe,
        problem.derivative_matvecs,
        problem.time,
        TIMING_STEPS,
    )
    rows: list[dict[str, float | str]] = []
    for repeat in range(TIMING_REPEATS):
        if repeat % 2 == 0:
            dense_started = perf_counter()
            dense_probability, dense_gradient = _dense_frechet_from_arrays(
                problem, hamiltonian, directions
            )
            dense_seconds = perf_counter() - dense_started
            direct_started = perf_counter()
            direct = direct_tridiagonal_replay_gradient(
                problem.matvec,
                problem.probe,
                problem.derivative_matvecs,
                problem.time,
                TIMING_STEPS,
            )
            direct_seconds = perf_counter() - direct_started
        else:
            direct_started = perf_counter()
            direct = direct_tridiagonal_replay_gradient(
                problem.matvec,
                problem.probe,
                problem.derivative_matvecs,
                problem.time,
                TIMING_STEPS,
            )
            direct_seconds = perf_counter() - direct_started
            dense_started = perf_counter()
            dense_probability, dense_gradient = _dense_frechet_from_arrays(
                problem, hamiltonian, directions
            )
            dense_seconds = perf_counter() - dense_started
        rows.append(
            {
                "matrix": problem.name,
                "dimension": float(problem.dimension),
                "repeat": float(repeat),
                "steps": float(TIMING_STEPS),
                "dense_frechet_seconds": float(dense_seconds),
                "direct_replay_seconds": float(direct_seconds),
                "dense_over_direct": float(dense_seconds / direct_seconds),
                "probability_absolute_error": float(
                    abs(direct.probability - dense_probability)
                ),
                "gradient_l2_error": float(
                    np.linalg.norm(direct.gradient - dense_gradient)
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _line_direct_vs_forward(
    rows: list[dict[str, float | str]], path: Path
) -> None:
    med_direct = []
    med_forward = []
    for depth in DEPTHS:
        selected = [row for row in rows if int(row["steps"]) == depth]
        med_direct.append(
            float(np.median([float(row["direct_replay_error"]) for row in selected]))
        )
        med_forward.append(
            float(np.median([float(row["forward_only_error"]) for row in selected]))
        )
    figure, axis = plt.subplots(figsize=(4.35, 3.15), constrained_layout=True)
    axis.semilogy(DEPTHS, med_direct, "o-", label="direct tridiagonal replay")
    axis.semilogy(DEPTHS, med_forward, "s--", label="reused-basis forward derivative")
    axis.set_xlabel("Lanczos steps $m$")
    axis.set_ylabel("median error against centered difference")
    axis.set_title("Finite-objective derivative on four PARSEC matrices")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(figure)


def _line_bound(
    rows: list[dict[str, float | str]], path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(4.35, 3.15), constrained_layout=True)
    steps = [float(row["steps"]) for row in rows]
    observed = np.maximum(
        [float(row["observed_gradient_l2_error"]) for row in rows],
        np.finfo(float).eps,
    )
    bound = [float(row["gradient_error_bound"]) for row in rows]
    axis.semilogy(steps, observed, "o-", label="observed error")
    axis.semilogy(steps, bound, "s--", label="exact-arithmetic upper bound")
    axis.set_xlabel("Lanczos steps $m$")
    axis.set_ylabel(r"gradient $\ell_2$ error")
    axis.set_title("Unprojected gradient bound on Si2")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(figure)


def _bar_storage(
    rows: list[dict[str, float | str]], path: Path
) -> None:
    labels = [str(row["matrix"]) for row in rows]
    full = [float(row["full_basis_mib"]) for row in rows]
    direct = [float(row["direct_replay_mib"]) for row in rows]
    positions = np.arange(len(rows))
    width = 0.38
    figure, axis = plt.subplots(figsize=(4.8, 3.15), constrained_layout=True)
    axis.bar(positions - width / 2, full, width, label="stored Lanczos basis")
    axis.bar(positions + width / 2, direct, width, label="direct replay")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=15, ha="right")
    axis.set_ylabel("large-object storage (MiB)")
    axis.set_title("Measured allocation ledger at $m=48$")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(figure)


def _bar_timing(
    rows: list[dict[str, float | str]], path: Path
) -> None:
    dense = np.asarray([float(row["dense_frechet_seconds"]) for row in rows])
    direct = np.asarray([float(row["direct_replay_seconds"]) for row in rows])
    medians = [float(np.median(dense)), float(np.median(direct))]
    figure, axis = plt.subplots(figsize=(4.2, 3.15), constrained_layout=True)
    bars = axis.bar(
        np.arange(2), medians, color=("0.55", "tab:blue"), edgecolor="black"
    )
    axis.set_yscale("log")
    axis.set_xticks(np.arange(2))
    axis.set_xticklabels(("dense Frechet", "direct replay"))
    axis.set_ylabel("median gradient time (s)")
    axis.set_title(f"Si2: {medians[0] / medians[1]:.0f}x median ratio")
    axis.grid(True, axis="y", which="both", alpha=0.25)
    for bar, value in zip(bars, medians):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_key(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="PARSEC download cache",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("code/results")
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("code/manuscript_assets/figures"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    problems = {
        name: build_real_problem(name, cache_dir=args.cache_dir, verify=True)
        for name in PARSEC_MATRICES
    }
    data_rows: list[dict[str, float | str]] = []
    for name, problem in problems.items():
        matrix, observed_hash = download_matrix(
            name, cache_dir=args.cache_dir, verify=True
        )
        data_rows.append(
            {
                "matrix": name,
                "dimension": float(problem.dimension),
                "nonzeros": float(matrix.nnz),
                "sha256": observed_hash,
                "source_url": PARSEC_MATRICES[name].url,
            }
        )

    cross_rows = run_cross_matrix_study(problems)
    si2 = problems["Si2"]
    dense_hamiltonian, dense_directions = _dense_arrays(si2)
    _, dense_gradient = _dense_frechet_from_arrays(
        si2, dense_hamiltonian, dense_directions
    )
    bound_rows = run_bound_study(si2, dense_gradient)
    storage_rows = run_storage_study(problems)
    timing_rows = run_timing_study(
        si2, dense_hamiltonian, dense_directions
    )

    outputs = {
        "data_manifest.csv": data_rows,
        "direct_cross_matrix.csv": cross_rows,
        "direct_bound_sweep.csv": bound_rows,
        "direct_storage.csv": storage_rows,
        "direct_timing.csv": timing_rows,
    }
    for filename, rows in outputs.items():
        _write_csv(args.output_dir / filename, rows)

    _line_direct_vs_forward(
        cross_rows, args.figure_dir / "direct_vs_forward"
    )
    _line_bound(bound_rows, args.figure_dir / "direct_bound")
    _bar_storage(storage_rows, args.figure_dir / "direct_storage")
    _bar_timing(timing_rows, args.figure_dir / "direct_timing")

    hypotheses = {
        "direct_error_below_forward_error_for_every_matrix_and_depth": all(
            float(row["direct_replay_error"])
            < float(row["forward_only_error"])
            for row in cross_rows
        ),
        "maximum_direct_centered_difference_error_below_2e_9": max(
            float(row["direct_replay_error"]) for row in cross_rows
        )
        < 2.0e-9,
        "unprojected_bound_covers_every_tested_depth": all(
            float(row["bound_covered"]) == 1.0 for row in bound_rows
        ),
        "direct_storage_below_full_basis_for_every_matrix": all(
            float(row["direct_replay_mib"]) < float(row["full_basis_mib"])
            for row in storage_rows
        ),
        "median_dense_time_exceeds_direct_time_by_50": (
            float(
                np.median(
                    [float(row["dense_frechet_seconds"]) for row in timing_rows]
                )
            )
            / float(
                np.median(
                    [float(row["direct_replay_seconds"]) for row in timing_rows]
                )
            )
            > 50.0
        ),
    }
    locked = {
        "schema_version": 2,
        "title": "Low-Memory Matrix-Free Gradients for Quantum Error Calibration",
        "algorithm": "direct tridiagonal-sensitivity replay",
        "data_source": "SuiteSparse Matrix Collection, PARSEC group",
        "depths": list(DEPTHS),
        "bound_depths": list(BOUND_DEPTHS),
        "finite_difference_step": FINITE_DIFFERENCE_STEP,
        "hypotheses": hypotheses,
        "data": data_rows,
        "cross_matrix": cross_rows,
        "bound_sweep": bound_rows,
        "storage": storage_rows,
        "timing": timing_rows,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    }
    locked_path = args.output_dir / "locked_results.json"
    locked_path.write_text(
        json.dumps(locked, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    tracked = [args.output_dir / name for name in outputs]
    tracked.extend(
        args.figure_dir / f"{stem}.{suffix}"
        for stem in (
            "direct_vs_forward",
            "direct_bound",
            "direct_storage",
            "direct_timing",
        )
        for suffix in ("pdf", "png")
    )
    tracked.append(locked_path)
    manifest = {
        "schema_version": 2,
        "timing_is_host_dependent": True,
        "files": {
            _manifest_key(path): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in tracked
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"hypotheses": hypotheses}, indent=2, sort_keys=True))
    if not all(hypotheses.values()):
        raise SystemExit("one or more predeclared hypotheses failed")


if __name__ == "__main__":
    main()
