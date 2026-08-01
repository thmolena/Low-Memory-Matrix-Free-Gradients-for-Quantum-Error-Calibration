#!/usr/bin/env python3
"""Validate the manuscript, evidence, and public-release contract."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
ASSETS = CODE / "manuscript_assets"
RESULTS = CODE / "results"
EXPECTED_FIGURES = {
    "code/manuscript_assets/figures/direct_vs_forward.pdf": "line",
    "code/manuscript_assets/figures/direct_bound.pdf": "line",
    "code/manuscript_assets/figures/direct_storage.pdf": "bar",
    "code/manuscript_assets/figures/direct_timing.pdf": "bar",
}
EXPECTED_RESULTS = {
    "data_manifest.csv",
    "direct_bound_sweep.csv",
    "direct_cross_matrix.csv",
    "direct_storage.csv",
    "direct_timing.csv",
    "locked_results.json",
    "manifest.json",
}


def fail(message: str) -> None:
    raise AssertionError(message)


def citation_keys(text: str) -> set[str]:
    found: set[str] = set()
    for match in re.finditer(r"\\cite\{([^}]*)\}", text, flags=re.DOTALL):
        found.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return found


def pdf_pages() -> int:
    output = subprocess.check_output(
        ["pdfinfo", str(ROOT / "main.pdf")],
        text=True,
        stderr=subprocess.STDOUT,
    )
    match = re.search(r"^Pages:\s+(\d+)$", output, flags=re.MULTILINE)
    if match is None:
        fail("pdfinfo did not report a page count")
    return int(match.group(1))


def main() -> int:
    expected_root = {
        ".gitignore", "LICENSE", "README.md", "code",
        "index.html", "main.pdf", "main.tex",
    }
    root_names = {path.name for path in ROOT.iterdir() if path.name != ".git"}
    if root_names != expected_root:
        fail(f"root contract drift: {sorted(root_names ^ expected_root)}")

    required = [
        ROOT / "main.tex",
        ROOT / "main.pdf",
        ROOT / "README.md",
        ROOT / "index.html",
        ASSETS / "main.bbl",
        ASSETS / "references.bib",
        CODE / "pyproject.toml",
        CODE / "README.md",
        CODE / "data" / "checksums.sha256",
        CODE / "data" / "THIRD_PARTY.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        fail(f"missing required files: {missing}")
    result_names = {path.name for path in RESULTS.iterdir() if path.is_file()}
    if result_names != EXPECTED_RESULTS:
        fail(f"result inventory drift: {sorted(result_names ^ EXPECTED_RESULTS)}")

    text = (ROOT / "main.tex").read_text(encoding="utf-8")
    bib = (ASSETS / "references.bib").read_text(encoding="utf-8")
    bbl = (ASSETS / "main.bbl").read_text(encoding="utf-8")

    class_match = re.search(r"\\documentclass\[([^]]+)\]\{revtex4-2\}", text)
    if class_match is None or "reprint" not in class_match.group(1).split(","):
        fail("manuscript is not a two-column REVTeX reprint")
    options = {item.strip().lower() for item in class_match.group(1).split(",")}
    if options != {"reprint", "superscriptaddress", "floatfix", "longbibliography"}:
        fail(f"unexpected document-class options: {sorted(options)}")
    for required_text in (
        r"\title{Low-Memory Matrix-Free Gradients for Quantum Error Calibration}",
        r"\author{Molena Huynh}",
        r"\email{molena.huynh@jmp.com}",
        r"\affiliation{North Carolina State University}",
    ):
        if required_text not in text:
            fail(f"missing manuscript metadata: {required_text}")

    figure_paths = re.findall(
        r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text
    )
    if set(figure_paths) != set(EXPECTED_FIGURES) or len(figure_paths) != 4:
        fail(f"unexpected manuscript figure set: {figure_paths}")
    for relative in EXPECTED_FIGURES:
        if not (ROOT / relative).is_file():
            fail(f"missing manuscript figure: {relative}")
        png = (ROOT / relative).with_suffix(".png")
        if not png.is_file():
            fail(f"missing website figure: {png.relative_to(ROOT)}")
    for suffix in (".pdf", ".png"):
        actual = {
            path.name for path in (ASSETS / "figures").glob(f"*{suffix}")
        }
        expected = {
            Path(path).with_suffix(suffix).name for path in EXPECTED_FIGURES
        }
        if actual != expected:
            fail(f"unexpected {suffix} figure inventory: {sorted(actual ^ expected)}")

    bib_keys = re.findall(r"@\w+\{([^,\s]+)", bib)
    if len(bib_keys) < 60 or len(bib_keys) != len(set(bib_keys)):
        fail(f"bibliography has {len(bib_keys)} entries or duplicate keys")
    cited = citation_keys(text)
    if cited != set(bib_keys):
        fail(
            f"citation mismatch: missing={sorted(cited-set(bib_keys))}, "
            f"uncited={sorted(set(bib_keys)-cited)}"
        )
    resolved = len(re.findall(r"^\\bibitem", bbl, flags=re.MULTILINE))
    if resolved != len(bib_keys):
        fail(f"resolved bibliography has {resolved} items for {len(bib_keys)} entries")

    table_count = len(re.findall(r"\\begin\{table\*?\}", text))
    figure_count = len(re.findall(r"\\begin\{figure\*?\}", text))
    theorem_count = len(
        re.findall(r"\\begin\{(?:theorem|proposition|corollary|lemma)\}", text)
    )
    if table_count != 5 or figure_count != 4:
        fail(f"unexpected numerical object counts: tables={table_count}, figures={figure_count}")
    if theorem_count < 15:
        fail(f"only {theorem_count} formal results")
    if len(text.split()) < 13_000:
        fail("manuscript is too short for the full-length contract")

    pages = pdf_pages()
    if not 25 <= pages <= 30:
        fail(f"compiled manuscript has {pages} pages, expected 25--30")

    locked = json.loads((RESULTS / "locked_results.json").read_text())
    if locked.get("algorithm") != "direct tridiagonal-sensitivity replay":
        fail("locked algorithm identity drift")
    hypotheses = locked.get("hypotheses", {})
    if len(hypotheses) != 5 or not all(hypotheses.values()):
        fail(f"locked hypotheses failed or incomplete: {hypotheses}")
    data = locked.get("data", [])
    if {row["matrix"] for row in data} != {"Si2", "SiH4", "benzene", "Si5H12"}:
        fail("real-data matrix inventory drift")
    for row in data:
        if not row["source_url"].startswith("https://") or len(row["sha256"]) != 64:
            fail(f"invalid provenance row: {row['matrix']}")
    if len(locked.get("cross_matrix", [])) != 24:
        fail("cross-matrix study does not contain 24 rows")

    hazards: list[str] = []
    forbidden_names = {
        ".DS_Store", ".pytest_cache", "__pycache__", "build", "dist",
        ".venv", ".idea", ".vscode",
    }
    forbidden_suffixes = {".aux", ".blg", ".log", ".out", ".xdv", ".pyc"}
    for path in ROOT.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.name in forbidden_names or path.name.endswith(".egg-info"):
            hazards.append(str(path.relative_to(ROOT)))
        if any(path.name.endswith(suffix) for suffix in forbidden_suffixes):
            hazards.append(str(path.relative_to(ROOT)))
    if hazards:
        fail(f"public-release debris: {sorted(set(hazards))}")

    print(json.dumps({
        "status": "PASS",
        "pages": pages,
        "references": len(bib_keys),
        "formal_results": theorem_count,
        "figures": {"line": 2, "bar": 2, "other": 0},
        "tables": table_count,
        "real_matrices": len(data),
        "hypotheses": len(hypotheses),
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
