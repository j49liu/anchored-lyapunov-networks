#!/usr/bin/env python3
"""Numerically screen the frozen candidates without any training imports.

The planar ``paper`` profile replays the independent fixed-level Sobol screen.
The 10D profile replays the full-dimensional IID radial test and a sampled box
boundary test.  Numerical screens are evidence, not formal certificates; use
``verify_planar.py`` for the planar proofs and ``verify_planar_crown.py`` for
corroborating bound propagation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from frozen_model import load_candidate
from systems import DOMINANT_DEGREES, PLANAR_DOMAINS, SYSTEMS


HERE = Path(__file__).resolve().parent


def sobol_box(count: int, bounds, seed: int, *, dtype=torch.float64):
    unit = torch.quasirandom.SobolEngine(
        len(bounds), scramble=True, seed=int(seed)
    ).draw(int(count), dtype=dtype)
    lower = torch.tensor([item[0] for item in bounds], dtype=dtype)
    upper = torch.tensor([item[1] for item in bounds], dtype=dtype)
    return unit * (upper - lower) + lower


def planar_boundary(count: int, bounds, seed: int):
    """Match the retained per-face Sobol construction."""

    per_face = max(2, math.ceil(count / (2 * len(bounds))))
    faces = []
    offset = 0
    for coordinate in range(len(bounds)):
        for side in bounds[coordinate]:
            points = sobol_box(per_face, bounds, seed + offset, dtype=torch.float32)
            points[:, coordinate] = side
            faces.append(points)
            offset += 1
    return torch.cat(faces)[:count].double()


def evaluate_batches(name, model, points, *, device, batch_size, margin):
    model = model.to(device=device, dtype=torch.float64).eval()
    flow_function = SYSTEMS[name]
    degree = DOMINANT_DEGREES[name]
    inside = 0
    violations = 0
    strict_violations = 0
    maximum_phi = -math.inf
    nonfinite = 0
    for batch in points.split(batch_size):
        states = batch.to(device=device, dtype=torch.float64)
        values = model.latent(states).reshape(-1)
        gradients = model.log_gradient(states)
        flow = flow_function(states)
        radius = torch.linalg.vector_norm(states, dim=1).clamp_min(1e-300)
        phi = (gradients * flow).sum(dim=1) / radius ** (degree - 1)
        finite = torch.isfinite(values) & torch.isfinite(phi)
        selected = finite & (values <= model.latent_level)
        inside += int(selected.sum())
        violations += int((selected & (phi >= -float(margin))).sum())
        strict_violations += int((selected & (phi >= 0)).sum())
        nonfinite += int((~finite).sum())
        if selected.any():
            maximum_phi = max(maximum_phi, float(phi[selected].max()))
    return {
        "points": len(points),
        "inside": inside,
        "coverage": inside / len(points),
        "margin": float(margin),
        "margin_violations_inside": violations,
        "strict_violations_inside": strict_violations,
        "maximum_scaled_log_derivative_inside": maximum_phi,
        "nonfinite_evaluations": nonfinite,
    }


def planar_screen(name, profile, device, batch_size):
    model = load_candidate(name)
    points = 2**20 if profile == "paper" else 2**14
    boundary_points = 65536 if profile == "paper" else 4096
    interior = sobol_box(points, PLANAR_DOMAINS[name], 20261021)
    result = evaluate_batches(
        name, model, interior, device=device, batch_size=batch_size, margin=0.01
    )
    boundary = planar_boundary(boundary_points, PLANAR_DOMAINS[name], 20261022)
    model = model.to(device=device, dtype=torch.float64)
    minimum = math.inf
    inside_boundary = 0
    with torch.no_grad():
        for batch in boundary.split(batch_size):
            values = model.latent(batch.to(device=device, dtype=torch.float64)).reshape(-1)
            minimum = min(minimum, float(values.min()))
            inside_boundary += int((values <= model.latent_level).sum())
    result["sampled_boundary"] = {
        "points": len(boundary),
        "minimum_H": minimum,
        "points_inside_or_on_level": inside_boundary,
        "contained": inside_boundary == 0 and minimum > model.latent_level,
    }
    result["fixed_latent_level"] = model.latent_level
    result["fixed_bounded_level"] = model.bounded_level
    return result


def iid_radial_points(count: int, radius: float, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    directions = torch.randn((count, 10), generator=generator, dtype=torch.float64)
    directions /= directions.abs().max(dim=1, keepdim=True).values
    radial = torch.rand((count, 1), generator=generator, dtype=torch.float64)
    return float(radius) * radial * directions


def logarithmic_radial_points(
    count: int, minimum_radius: float, maximum_radius: float, seed: int
):
    """Sample directions uniformly and radii uniformly on a logarithmic scale."""

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    directions = torch.randn((count, 10), generator=generator, dtype=torch.float64)
    directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    unit = torch.rand((count, 1), generator=generator, dtype=torch.float64)
    radii = torch.exp(
        math.log(float(minimum_radius))
        + unit * math.log(float(maximum_radius) / float(minimum_radius))
    )
    return radii * directions


def grune_boundary(count: int, radius: float, seed: int):
    if count < 20:
        raise ValueError("the boundary screen must include all 20 face centers")
    unit = torch.quasirandom.SobolEngine(
        11, scramble=True, seed=int(seed)
    ).draw(int(count), dtype=torch.float64)
    points = (2.0 * unit[:, 1:] - 1.0) * float(radius)
    face = torch.floor(unit[:, 0] * 20).to(torch.int64)
    coordinate = torch.div(face, 2, rounding_mode="floor")
    sign = 2.0 * (face % 2).to(torch.float64) - 1.0
    points[torch.arange(count), coordinate] = sign * float(radius)
    for coordinate_index in range(10):
        for side_index, side in enumerate((-1.0, 1.0)):
            row = 2 * coordinate_index + side_index
            points[row].zero_()
            points[row, coordinate_index] = side * float(radius)
    return points


def grune_screen(profile, device, batch_size):
    model = load_candidate("grune10d")
    points = 2**20 if profile == "paper" else 2**14
    boundary_points = 2**18 if profile == "paper" else 4096
    near_origin_points = 2**18 if profile == "paper" else 2**13
    # This seed was reserved until the homogeneous-core checkpoint, gain,
    # and level were frozen.  It matches the retained independent evaluation.
    population = iid_radial_points(points, 40.0, 20262111)
    result = evaluate_batches(
        "grune10d",
        model,
        population,
        device=device,
        batch_size=batch_size,
        margin=0.01,
    )
    near_origin = evaluate_batches(
        "grune10d",
        model,
        logarithmic_radial_points(
            near_origin_points, 1e-12, 1e-3, 20262114
        ),
        device=device,
        batch_size=batch_size,
        margin=0.01,
    )
    near_origin["minimum_radius"] = 1e-12
    near_origin["maximum_radius"] = 1e-3
    near_origin["proposal"] = (
        "Gaussian directions normalized in Euclidean norm; radius uniform on a logarithmic scale"
    )
    boundary = grune_boundary(boundary_points, 40.0, 20262113)
    model = model.to(device=device, dtype=torch.float64)
    minimum = math.inf
    inside_boundary = 0
    with torch.no_grad():
        for batch in boundary.split(batch_size):
            values = model.latent(batch.to(device=device, dtype=torch.float64)).reshape(-1)
            minimum = min(minimum, float(values.min()))
            inside_boundary += int((values <= model.latent_level).sum())
    result["proposal"] = "IID Gaussian directions normalized in infinity norm; uniform radial fraction"
    result["near_origin_log_radial"] = near_origin
    result["sampled_boundary"] = {
        "points": len(boundary),
        "minimum_H": minimum,
        "points_inside_or_on_level": inside_boundary,
        "contained": inside_boundary == 0 and minimum > model.latent_level,
    }
    result["fixed_latent_level"] = model.latent_level
    result["fixed_bounded_level"] = model.bounded_level
    result["claim_scope"] = "empirical post-selection screen; not formal verification"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    requested = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if requested == "auto":
        requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    results = {
        "schema": "frozen-candidate-numerical-screen-v1",
        "profile": args.profile,
        "device": str(device),
        "candidates": {
            "two_machine": planar_screen("two_machine", args.profile, device, args.batch_size),
            "cubic": planar_screen("cubic", args.profile, device, args.batch_size),
            "grune10d": grune_screen(args.profile, device, args.batch_size),
        },
    }
    text = json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
