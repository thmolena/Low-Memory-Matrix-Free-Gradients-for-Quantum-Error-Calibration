"""Authenticated PARSEC study for parameter-blocked tangent replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from .block_replay import parameter_blocked_replay_gradient
from .hamiltonians import PARSEC_MATRICES, build_real_problem
from .low_memory import direct_tridiagonal_replay_gradient

MATRICES = ("Si2", "SiH4", "benzene", "Si5H12")
DEPTHS = (8, 16, 32, 48)
TIMING_STEPS = 32
TIMING_REPEATS = 9
PARAMETER_COUNTS = (1, 2, 3, 4, 6, 8)


def _timed(callable_) -> tuple[object, float]:
    start = time.perf_counter()
    result = callable_()
    return result, time.perf_counter() - start


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "median_seconds": float(np.median(array)),
        "lower_seconds": float(np.quantile(array, 0.1)),
        "upper_seconds": float(np.quantile(array, 0.9)),
    }


def run_study() -> dict[str, object]:
    accuracy_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    data_rows: list[dict[str, object]] = []

    for name in MATRICES:
        problem = build_real_problem(name, time=8.0)
        spec = PARSEC_MATRICES[name]
        data_rows.append(
            {
                "matrix": name,
                "dimension": problem.dimension,
                "sha256": problem.sha256,
                "source_url": spec.url,
            }
        )
        for steps in DEPTHS:
            scalar = direct_tridiagonal_replay_gradient(
                problem.matvec,
                problem.probe,
                problem.derivative_matvecs,
                problem.time,
                steps,
            )
            blocked = parameter_blocked_replay_gradient(
                problem.matvec,
                problem.probe,
                problem.derivative_matvecs,
                problem.time,
                steps,
                parameter_block_size=len(problem.derivative_matvecs),
            )
            difference = blocked.gradient - scalar.gradient
            accuracy_rows.append(
                {
                    "matrix": name,
                    "dimension": problem.dimension,
                    "steps": steps,
                    "gradient_l2_difference": float(np.linalg.norm(difference)),
                    "gradient_max_difference": float(np.max(np.abs(difference))),
                    "probability_difference": float(
                        abs(blocked.probability - scalar.probability)
                    ),
                    "scalar_hamiltonian_launches": scalar.hamiltonian_matvecs,
                    "blocked_hamiltonian_launches": (
                        blocked.hamiltonian_vector_calls
                        + blocked.hamiltonian_block_calls
                    ),
                    "blocked_column_equivalents": blocked.hamiltonian_column_equivalents,
                    "derivative_vector_calls": blocked.derivative_vector_calls,
                    "blocked_peak_vector_equivalents": blocked.peak_ambient_vector_equivalents,
                }
            )

        parameter_count = len(problem.derivative_matvecs)
        direct_tridiagonal_replay_gradient(
            problem.matvec,
            problem.probe,
            problem.derivative_matvecs,
            problem.time,
            TIMING_STEPS,
        )
        parameter_blocked_replay_gradient(
            problem.matvec,
            problem.probe,
            problem.derivative_matvecs,
            problem.time,
            TIMING_STEPS,
            parameter_block_size=parameter_count,
        )
        scalar_times: list[float] = []
        blocked_times: list[float] = []
        comparison_errors: list[float] = []
        for _ in range(TIMING_REPEATS):
            scalar, scalar_elapsed = _timed(
                lambda: direct_tridiagonal_replay_gradient(
                    problem.matvec,
                    problem.probe,
                    problem.derivative_matvecs,
                    problem.time,
                    TIMING_STEPS,
                )
            )
            blocked, blocked_elapsed = _timed(
                lambda: parameter_blocked_replay_gradient(
                    problem.matvec,
                    problem.probe,
                    problem.derivative_matvecs,
                    problem.time,
                    TIMING_STEPS,
                    parameter_block_size=parameter_count,
                )
            )
            scalar_times.append(scalar_elapsed)
            blocked_times.append(blocked_elapsed)
            comparison_errors.append(
                float(np.linalg.norm(blocked.gradient - scalar.gradient))
            )
        scalar_summary = _quantiles(scalar_times)
        blocked_summary = _quantiles(blocked_times)
        timing_rows.append(
            {
                "matrix": name,
                "dimension": problem.dimension,
                "steps": TIMING_STEPS,
                "repeats": TIMING_REPEATS,
                "scalar": scalar_summary,
                "blocked": blocked_summary,
                "median_speedup": (
                    scalar_summary["median_seconds"]
                    / blocked_summary["median_seconds"]
                ),
                "maximum_gradient_difference": max(comparison_errors),
            }
        )

    scaling_rows: list[dict[str, object]] = []
    for parameter_count in PARAMETER_COUNTS:
        theta = tuple(
            0.04 * np.cos(np.arange(parameter_count, dtype=float) + 0.5)
        )
        problem = build_real_problem("SiH4", time=8.0, theta_true=theta)
        scalar_times: list[float] = []
        blocked_times: list[float] = []
        blocked_result = None
        scalar_result = None
        for _ in range(5):
            scalar_result, elapsed = _timed(
                lambda: direct_tridiagonal_replay_gradient(
                    problem.matvec,
                    problem.probe,
                    problem.derivative_matvecs,
                    problem.time,
                    24,
                )
            )
            scalar_times.append(elapsed)
            blocked_result, elapsed = _timed(
                lambda: parameter_blocked_replay_gradient(
                    problem.matvec,
                    problem.probe,
                    problem.derivative_matvecs,
                    problem.time,
                    24,
                    parameter_block_size=parameter_count,
                )
            )
            blocked_times.append(elapsed)
        assert blocked_result is not None and scalar_result is not None
        scaling_rows.append(
            {
                "parameter_count": parameter_count,
                "steps": 24,
                "scalar_launches": scalar_result.hamiltonian_matvecs,
                "blocked_launches": (
                    blocked_result.hamiltonian_vector_calls
                    + blocked_result.hamiltonian_block_calls
                ),
                "scalar_peak_vector_equivalents": float(
                    scalar_result.estimated_peak_vector_equivalents
                ),
                "blocked_peak_vector_equivalents": (
                    blocked_result.peak_ambient_vector_equivalents
                ),
                "scalar": _quantiles(scalar_times),
                "blocked": _quantiles(blocked_times),
                "gradient_l2_difference": float(
                    np.linalg.norm(blocked_result.gradient - scalar_result.gradient)
                ),
            }
        )

    semantic_core: dict[str, object] = {
        "schema_version": "2.0",
        "algorithm": "parameter-blocked tridiagonal-sensitivity replay",
        "data": data_rows,
        "accuracy": accuracy_rows,
        "ledger": [
            {
                "parameter_count": row["parameter_count"],
                "steps": row["steps"],
                "scalar_launches": row["scalar_launches"],
                "blocked_launches": row["blocked_launches"],
                "scalar_peak_vector_equivalents": row[
                    "scalar_peak_vector_equivalents"
                ],
                "blocked_peak_vector_equivalents": row[
                    "blocked_peak_vector_equivalents"
                ],
                "gradient_l2_difference": row["gradient_l2_difference"],
            }
            for row in scaling_rows
        ],
        "checks": {
            "all_gradient_differences_below_2e_12": all(
                row["gradient_l2_difference"] < 2.0e-12
                for row in accuracy_rows
            ),
            "all_probability_differences_below_1e_14": all(
                row["probability_difference"] < 1.0e-14
                for row in accuracy_rows
            ),
            "all_full_blocks_use_three_m_launches": all(
                row["blocked_hamiltonian_launches"] == 3 * row["steps"]
                for row in accuracy_rows
            ),
            "all_scalar_runs_use_seven_m_launches": all(
                row["scalar_hamiltonian_launches"] == 7 * row["steps"]
                for row in accuracy_rows
            ),
        },
    }
    payload = json.dumps(
        semantic_core, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {
        **semantic_core,
        "semantic_sha256": hashlib.sha256(payload).hexdigest(),
        "timing": timing_rows,
        "parameter_scaling": scaling_rows,
        "timing_checks": {
            "blocked_faster_on_every_matrix": all(
                row["median_speedup"] > 1.0 for row in timing_rows
            ),
            "minimum_matrix_speedup": min(
                row["median_speedup"] for row in timing_rows
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }


def make_figures(result: dict[str, object], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
        }
    )
    colors = dict(zip(MATRICES, plt.cm.viridis(np.linspace(0.1, 0.9, 4))))

    fig, ax = plt.subplots(figsize=(5.4, 3.15))
    for matrix in MATRICES:
        rows = [row for row in result["accuracy"] if row["matrix"] == matrix]
        ax.semilogy(
            [row["steps"] for row in rows],
            [max(row["gradient_l2_difference"], 1e-18) for row in rows],
            "o-",
            label=matrix,
            color=colors[matrix],
        )
    ax.axhline(2e-12, color="#b22222", linestyle="--", label="registered tolerance")
    ax.set(xlabel="Lanczos steps", ylabel="blocked--scalar gradient difference")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"block_equivalence.{suffix}", bbox_inches="tight")
    plt.close(fig)

    rows = result["timing"]
    x = np.arange(len(rows))
    scalar = [row["scalar"]["median_seconds"] for row in rows]
    blocked = [row["blocked"]["median_seconds"] for row in rows]
    fig, ax = plt.subplots(figsize=(5.4, 3.15))
    ax.semilogy(x, scalar, "s--", label="parameterwise replay")
    ax.semilogy(x, blocked, "o-", label="full parameter block")
    for index, row in enumerate(rows):
        ax.text(index, blocked[index] * 1.18, f"{row['median_speedup']:.2f}x", ha="center", fontsize=8)
    ax.set_xticks(x, [row["matrix"] for row in rows])
    ax.set(ylabel="median gradient time (s)", xlabel="authenticated PARSEC matrix")
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"block_runtime.{suffix}", bbox_inches="tight")
    plt.close(fig)

    rows = result["parameter_scaling"]
    p = [row["parameter_count"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    axes[0].plot(p, [row["scalar_launches"] for row in rows], "s--", label="parameterwise")
    axes[0].plot(p, [row["blocked_launches"] for row in rows], "o-", label="full block")
    axes[0].set(xlabel="parameter count", ylabel="Hamiltonian launches")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(p, [row["scalar_peak_vector_equivalents"] for row in rows], "s--", label="parameterwise")
    axes[1].plot(p, [row["blocked_peak_vector_equivalents"] for row in rows], "o-", label="full block")
    axes[1].set(xlabel="parameter count", ylabel="peak vector equivalents")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"launch_memory_tradeoff.{suffix}", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.15))
    ax.plot(p, [row["scalar"]["median_seconds"] for row in rows], "s--", label="parameterwise")
    ax.plot(p, [row["blocked"]["median_seconds"] for row in rows], "o-", label="full block")
    ax.set(xlabel="parameter count", ylabel="median gradient time (s)")
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"parameter_runtime.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("code/results/block_replay")
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("code/manuscript_assets/figures"),
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result_path = args.output_dir / "block_replay_study.json"
    expected = None
    if args.verify and result_path.exists():
        expected = json.loads(result_path.read_text())["semantic_sha256"]
    result = run_study()
    if expected is not None and result["semantic_sha256"] != expected:
        raise SystemExit(
            f"semantic hash mismatch: {expected} != {result['semantic_sha256']}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    make_figures(result, args.figure_dir)
    print(json.dumps({
        "semantic_sha256": result["semantic_sha256"],
        "checks": result["checks"],
        "timing_checks": result["timing_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
