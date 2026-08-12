"""
Shared plumbing for the paper experiments.

Deliberately small: seeding, artifact paths, a CSV/JSON/markdown writer, a
figure helper, and a check-collector so each script can double as an
integration test. Anything cleverer than this belongs in the experiment that
needs it, not here.

Artifact formats are constrained by the repo's ``.gitignore``, which ignores
``*.png`` and ``*.svg`` globally: figures are written as **PDF**, which is also
what LaTeX wants. ``*.csv`` is ignored too, but ``paper/results/**`` is
explicitly re-included, so raw data can be committed.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
# ``HC_PAPER_RESULTS`` redirects artifacts elsewhere, so a smoke run (or the unit
# test that shells out to one of these scripts) cannot overwrite the committed
# results of a real run.
RESULTS = Path(
    os.environ.get("HC_PAPER_RESULTS", Path(__file__).resolve().parent / "results")
)


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #


def set_seed(seed: int = 0) -> None:
    """Seed every generator the experiments touch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def use_float64() -> None:
    """Run in double precision.

    Required wherever a solver's own tolerances matter: NonOpt's line search
    tests decreases around 1e-10, and the KKT residuals in E0a are compared
    against QP-solver output that is itself accurate to ~1e-9.
    """
    torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------- #
# figure style — reuse the repo's single style source
# --------------------------------------------------------------------------- #


def load_by_path(path, name: str = None):
    """Import a module from a file path.

    ``benchmark/new_bench/`` is not a package, so its modules cannot be imported
    normally -- but a few of them hold logic the paper experiments should not
    duplicate (the figure style, the fairness constraint definitions). Importing
    by path reuses that logic instead of letting two copies drift.
    """
    path = Path(path)
    name = name or f"_paper_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Register before executing so a module that imports itself by name (or is
    # imported twice) does not get two distinct copies.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_benchmark_module(stem: str):
    """Import ``benchmark/new_bench/<stem>.py`` by path."""
    return load_by_path(REPO_ROOT / "benchmark" / "new_bench" / f"{stem}.py")


def _load_plot_style():
    """The repo's declared single style source for all paper figures."""
    return load_by_path(
        REPO_ROOT / "benchmark" / "new_bench" / "plotting" / "plot_style.py",
        "_paper_plot_style",
    )


def figure(nrows=1, ncols=1, width=None, row_height=None, **kwargs):
    """A styled figure sized for the paper's text width.

    :return: ``(fig, axes, plt)`` — ``axes`` is always a flat list.
    """
    style = _load_plot_style()
    style.set_neurips_style()
    import matplotlib.pyplot as plt

    width = width if width is not None else style.TEXT_WIDTH
    row_height = row_height if row_height is not None else style.ROW_H
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width, row_height * nrows), **kwargs
    )
    axes = np.atleast_1d(np.asarray(axes)).ravel().tolist()
    return fig, axes, plt


def save_figure(fig, name: str, experiment: str) -> Path:
    """Write ``fig`` as a PDF under ``paper/results/<experiment>/``."""
    path = _dir(experiment) / f"{name}.pdf"
    fig.savefig(path)
    print(f"  figure -> {_display(path)}")
    return path


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #


def _dir(experiment: str) -> Path:
    path = RESULTS / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path


def _display(path: Path) -> str:
    """Repo-relative when possible; absolute when results were redirected."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_csv(rows: list[dict], name: str, experiment: str) -> Path:
    """Write records as CSV; the union of all keys becomes the header."""
    if not rows:
        raise ValueError(f"no rows to write for {name!r}")
    columns, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    path = _dir(experiment) / f"{name}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  csv    -> {_display(path)}  ({len(rows)} rows)")
    return path


def write_table(rows: list[dict], name: str, experiment: str, *,
                columns: list[str] = None, floatfmt: str = "{:.3e}",
                title: str = None) -> Path:
    """Write records as a Markdown table plus a machine-readable JSON sidecar."""
    columns = columns or list(rows[0])

    def cell(value):
        if isinstance(value, float):
            return floatfmt.format(value)
        return str(value)

    lines = []
    if title:
        lines += [f"# {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(c, "")) for c in columns) + " |")

    path = _dir(experiment) / f"{name}.md"
    path.write_text("\n".join(lines) + "\n")
    (_dir(experiment) / f"{name}.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"  table  -> {_display(path)}")
    return path


# --------------------------------------------------------------------------- #
# check collection
# --------------------------------------------------------------------------- #


class Checks:
    """Collects pass/fail expectations so a script can double as a test.

    Every experiment states its predictions up front and registers them here; a
    script run with ``--check`` exits non-zero if any prediction failed, which
    is what makes the experiments falsifiable rather than merely descriptive.

    A prediction the experiment has already falsified and *explained* is
    registered with ``known_false=<reason>``. It is still evaluated and still
    printed, so the record of having been wrong stays in the output, but it does
    not fail the gate — and if it ever starts passing, that inverts into a
    failure, since the recorded explanation would no longer describe reality.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        # (ok, claim, detail, known_false_reason)
        self.results: list[tuple[bool, str, str, Optional[str]]] = []

    def expect(self, ok: bool, claim: str, detail: str = "",
               known_false: str = None) -> bool:
        self.results.append((bool(ok), claim, detail, known_false))
        return bool(ok)

    def _marks(self):
        for ok, claim, detail, known_false in self.results:
            if known_false is None:
                mark = "PASS" if ok else "FAIL"
            elif ok:
                mark = "UNEXPECTED-PASS"
            else:
                mark = "KNOWN-FALSE"
            yield mark, ok, claim, detail, known_false

    def report(self) -> int:
        """Print every registered expectation; return a process exit code."""
        if not self.results:
            return 0
        failures = 0
        print("\nPredictions:")
        for mark, _, claim, detail, known_false in self._marks():
            failures += mark in ("FAIL", "UNEXPECTED-PASS")
            print(f"  [{mark}] {claim}")
            if detail:
                print(f"         {detail}")
            if known_false:
                print(f"         known false: {known_false}")
        if failures and self.enabled:
            print(f"\n{failures} of {len(self.results)} predictions failed.")
            return 1
        if failures:
            print(f"\n{failures} prediction(s) failed (--check not set, exiting 0).")
        return 0

    def write(self, experiment: str, name: str) -> Optional[Path]:
        """Persist the evaluated predictions alongside the numeric artifacts.

        Without this the predictions live only in a terminal scrollback, which
        makes the reproducibility bundle strictly less informative than the run:
        the tables say what happened, this says what was *expected* to happen.
        """
        if not self.results:
            return None
        lines = [f"# {experiment}: registered predictions", ""]
        for mark, _, claim, detail, known_false in self._marks():
            lines.append(f"- **{mark}** — {claim}")
            if detail:
                lines.append(f"  - {detail}")
            if known_false:
                lines.append(f"  - *known false:* {known_false}")
        path = _dir(experiment) / f"{name}.md"
        path.write_text("\n".join(lines) + "\n")
        print(f"  checks -> {_display(path)}")
        return path


def main_exit(checks: Checks, experiment: str = None, name: str = None) -> None:
    """Report checks, persist them if asked, and exit with the right status."""
    code = checks.report()
    if experiment and name:
        checks.write(experiment, name)
    sys.exit(code)
