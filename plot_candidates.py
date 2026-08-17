#!/usr/bin/env python3
"""Plot the three frozen candidates from their portable NumPy weights.

The left column shows a conservative four-significant-digit rendering of the
reported sublevel boundary on a two-dimensional domain or coordinate slice.
The right column shows the corresponding bounded candidate ``W = tanh(H)`` as
a surface.  No training checkpoint or training module is imported.

For the 10D example, all coordinates except ``x1`` and ``x9`` are fixed
to zero.  This is a visualization of a coordinate slice, not an invariant
subsystem or a formal certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from frozen_model import DEFAULT_MANIFEST, load_candidate, load_manifest


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "figures"
ORANGE = "#D55E00"
FORMAL_BLUE = "#0072B2"
EMPIRICAL_PURPLE = "#7B3294"
TEXT = "#17212B"
GRID = "#D7DEE5"
PANEL_SPECS = (
    {
        "name": "two_machine",
        "display": "Example VI.1 (Two-machine)",
        "coordinates": (0, 1),
        "labels": (r"$x_1$", r"$x_2$"),
        "slice_note": None,
    },
    {
        "name": "cubic",
        "display": "Example VI.2 (Cubic core)",
        "coordinates": (0, 1),
        "labels": (r"$x_1$", r"$x_2$"),
        "slice_note": None,
    },
    {
        "name": "grune10d",
        "display": "Example VI.3 (10D)",
        "coordinates": (0, 8),
        "labels": (r"$x_1$", r"$x_9$"),
        "slice_note": "all other states fixed to zero; coordinate slice is not invariant",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reported_level(value: float, significant_digits: int = 4) -> float:
    """Round a positive level downward to the reported precision."""

    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("reported levels must be finite and positive")
    exponent = math.floor(math.log10(value))
    scale = 10.0 ** (significant_digits - 1 - exponent)
    return math.floor(value * scale) / scale


def configure_style() -> None:
    # P052 is the URW Palatino-compatible face used here for MathPazo styling.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["P052", "Palatino"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "P052",
            "mathtext.it": "P052:italic",
            "mathtext.bf": "P052:bold",
            "mathtext.cal": "P052:italic",
            "mathtext.sf": "P052",
            "mathtext.fallback": "stix",
            "font.size": 11.2,
            "axes.titlesize": 12.2,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "axes.edgecolor": "#68737D",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "pdf.compression": 9,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def evidence_label(metadata: dict, manifest_path: Path) -> tuple[str, str]:
    """Return a concise label and color from hash-bound manifest evidence."""

    if metadata["dimension"] == 2:
        proof_relative = metadata["evidence"]["standalone_proof"]
        proof_path = manifest_path.parent / proof_relative
        expected = metadata["evidence"]["standalone_proof_sha256"]
        if sha256(proof_path) != expected:
            raise RuntimeError(f"formal-proof hash mismatch for {metadata['name']}")
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        ledger = proof.get("queries", [])
        unsat = sum(entry.get("result", "").lower() == "unsat" for entry in ledger)
        if proof.get("verified") is not True or not ledger or unsat != len(ledger):
            raise RuntimeError(f"incomplete formal evidence for {metadata['name']}")
        coverage = 100.0 * float(metadata["independent_coverage"])
        return (
            f"FORMAL dReal certificate · {unsat}/{len(ledger)} UNSAT\n"
            f"Independent box coverage {coverage:.2f}%",
            FORMAL_BLUE,
        )

    scope = metadata.get("evidence_scope", "").lower()
    if "not formal" not in scope:
        raise RuntimeError("10D evidence must be identified as non-formal")
    coverage = 100.0 * float(metadata["iid_radial_coverage"])
    return (
        "EMPIRICAL SCREEN ONLY · not formally verified\n"
        f"IID-radial coverage {coverage:.2f}% · non-invariant coordinate slice",
        EMPIRICAL_PURPLE,
    )


def make_states(
    dimension: int,
    first: int,
    second: int,
    first_grid: np.ndarray,
    second_grid: np.ndarray,
) -> torch.Tensor:
    states = torch.zeros((first_grid.size, dimension), dtype=torch.float64)
    states[:, first] = torch.from_numpy(first_grid.reshape(-1))
    states[:, second] = torch.from_numpy(second_grid.reshape(-1))
    return states


def bounded_grid(
    model,
    bounds,
    coordinates,
    points: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first, second = coordinates
    first_axis = np.linspace(bounds[first][0], bounds[first][1], points, dtype=np.float64)
    second_axis = np.linspace(bounds[second][0], bounds[second][1], points, dtype=np.float64)
    first_grid, second_grid = np.meshgrid(first_axis, second_axis)
    states = make_states(model.dimension, first, second, first_grid, second_grid)
    values = []
    model = model.to(dtype=torch.float64).eval()
    with torch.inference_mode():
        for batch in states.split(batch_size):
            values.append(model.bounded(batch).reshape(-1).cpu())
    value_grid = torch.cat(values).numpy().reshape(first_grid.shape)
    if not np.isfinite(value_grid).all():
        raise RuntimeError(f"nonfinite values while plotting {model.name}")
    return first_grid, second_grid, value_grid


def draw_level_panel(
    axis,
    first_grid,
    second_grid,
    values,
    metadata,
    spec,
):
    bounded_level = reported_level(float(metadata["bounded_level"]))
    heatmap = axis.contourf(
        first_grid,
        second_grid,
        values,
        levels=np.linspace(0.0, 1.0, 41),
        cmap="viridis",
        norm=colors.Normalize(0.0, 1.0),
        extend="max",
    )
    contour = axis.contour(
        first_grid,
        second_grid,
        values,
        levels=[bounded_level],
        colors=[ORANGE],
        linewidths=2.6,
        zorder=5,
    )
    if not contour.allsegs[0]:
        raise RuntimeError(f"reported level is absent from the {metadata['name']} slice")
    axis.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=58,
        color="white",
        edgecolor="#111111",
        linewidth=0.7,
        zorder=6,
    )
    first_label, second_label = spec["labels"]
    axis.set_xlabel(first_label)
    axis.set_ylabel(second_label)
    axis.set_aspect("equal", adjustable="box")
    level_kind = "Tested" if metadata["name"] == "grune10d" else "Verified"
    axis.set_title(
        f"{spec['display']}\n{level_kind} level set", color=TEXT, pad=8
    )
    axis.grid(color=GRID, linewidth=0.45, alpha=0.38)
    axis.set_axisbelow(True)
    axis.legend(
        handles=[Line2D([0], [0], color=ORANGE, linewidth=2.6)],
        labels=[rf"$W_\theta = {bounded_level:.4g}$"],
        loc="upper left" if metadata["name"] == "cubic" else "lower left",
        frameon=True,
        framealpha=0.86,
        facecolor="white",
        edgecolor="none",
        fontsize=9.5,
        handlelength=2.0,
        borderpad=0.35,
    )
    return heatmap


def draw_surface_panel(
    axis,
    first_grid,
    second_grid,
    values,
    metadata,
    spec,
):
    bounded_level = reported_level(float(metadata["bounded_level"]))
    surface = axis.plot_surface(
        first_grid,
        second_grid,
        values,
        cmap="viridis",
        norm=colors.Normalize(0.0, 1.0),
        rcount=values.shape[0],
        ccount=values.shape[1],
        linewidth=0.0,
        antialiased=True,
        shade=True,
    )
    axis.contour(
        first_grid,
        second_grid,
        values,
        levels=[bounded_level],
        zdir="z",
        offset=0.0,
        colors=[ORANGE],
        linewidths=2.5,
    )
    axis.scatter(
        [0.0],
        [0.0],
        [0.0],
        marker="*",
        s=36,
        color="#111111",
        depthshade=False,
    )
    first_label, second_label = spec["labels"]
    axis.set_xlabel(first_label, labelpad=5)
    axis.set_ylabel(second_label, labelpad=5)
    axis.set_zlabel(r"$W_\theta$", labelpad=4)
    axis.set_zlim(0.0, 1.0)
    axis.set_title(
        f"{spec['display']}\nLearned Lyapunov function", color=TEXT, pad=8
    )
    axis.view_init(elev=28.0, azim=-125.0)
    axis.set_box_aspect((1.0, 1.0, 0.66))
    axis.grid(True)
    return surface


def render(
    manifest_path: Path,
    output_dir: Path,
    grid_points: int,
    surface_points: int,
    batch_size: int,
) -> list[Path]:
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    configure_style()
    figure = plt.figure(figsize=(12.2, 13.6))
    grid = figure.add_gridspec(
        3,
        2,
        width_ratios=(1.0, 1.08),
        left=0.065,
        right=0.91,
        bottom=0.055,
        top=0.91,
        hspace=0.36,
        wspace=0.21,
    )
    surface = None
    for row, spec in enumerate(PANEL_SPECS):
        name = spec["name"]
        metadata = manifest["candidates"][name]
        model = load_candidate(name, manifest_path).to(dtype=torch.float64).eval()
        if not np.isclose(
            np.tanh(float(metadata["latent_level"])),
            float(metadata["bounded_level"]),
            rtol=0.0,
            atol=2e-15,
        ):
            raise RuntimeError(f"latent and bounded levels disagree for {name}")
        # Validate the evidence record even though the plot itself remains uncluttered.
        evidence_label(metadata, manifest_path)
        level_first, level_second, level_values = bounded_grid(
            model,
            metadata["domain"],
            spec["coordinates"],
            grid_points,
            batch_size,
        )
        surface_first, surface_second, surface_values = bounded_grid(
            model,
            metadata["domain"],
            spec["coordinates"],
            surface_points,
            batch_size,
        )
        level_axis = figure.add_subplot(grid[row, 0])
        surface_axis = figure.add_subplot(grid[row, 1], projection="3d")
        draw_level_panel(
            level_axis,
            level_first,
            level_second,
            level_values,
            metadata,
            spec,
        )
        surface = draw_surface_panel(
            surface_axis,
            surface_first,
            surface_second,
            surface_values,
            metadata,
            spec,
        )

    figure.suptitle(
        "Level sets and learned Lyapunov functions",
        x=0.49,
        y=0.985,
        fontsize=18.0,
        fontweight="bold",
        color=TEXT,
    )
    colorbar_axis = figure.add_axes((0.93, 0.20, 0.016, 0.60))
    figure.colorbar(surface, cax=colorbar_axis)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "frozen_candidates.pdf"
    png_path = output_dir / "frozen_candidates.png"
    figure.savefig(
        pdf_path,
        dpi=300,
        facecolor="white",
        metadata={
            "Title": "Level sets and learned Lyapunov functions",
            "Author": "LYZNet",
            "Creator": Path(__file__).name,
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        png_path,
        dpi=240,
        facecolor="white",
        metadata={
            "Title": "Level sets and learned Lyapunov functions",
            "Author": "LYZNet",
            "Software": Path(__file__).name,
        },
    )
    plt.close(figure)
    return [pdf_path, png_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-points", type=int, default=401)
    parser.add_argument("--surface-points", type=int, default=151)
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args()
    if args.grid_points < 51 or args.surface_points < 31:
        parser.error("grid-points must be at least 51 and surface-points at least 31")
    outputs = render(
        args.manifest,
        args.output_dir.resolve(),
        args.grid_points,
        args.surface_points,
        args.batch_size,
    )
    print(
        json.dumps(
            {
                "source": "portable .npz weights only",
                "outputs": [
                    {
                        "path": str(path),
                        "sha256": sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in outputs
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
