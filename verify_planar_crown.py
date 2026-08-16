#!/usr/bin/env python3
"""GPU CROWN replay for the two frozen planar Lyapunov certificates.

The script uses the public Python API from the alpha-beta-CROWN repository,
but it does not depend on LyZNet or on training code.  It proves four facts:

1. the homogeneous core decreases on the complete unit circle;
2. the retained analytic perturbation bound closes on ``||x|| <= rho``;
3. on the rest of the selected sublevel inside the verification box, the
   scaled logarithmic derivative is strictly negative; and
4. the selected sublevel does not meet any face of the box.

The installed alpha-beta-CROWN release does not support the polar equality
``||u|| = 1`` or the removable singularity at the origin as input
constraints.  We therefore use exact parameterizations and exhaustive input
subdivision:

* ``u = (cos(t), sin(t))`` covers the unit circle;
* an inner square is chosen strictly inside the analytic local ball; and
* four rectangles cover the complement of that square in the box.

Every terminal tile is accepted only when a certified CROWN lower bound
proves at least one disjunct of the desired implication.  Unresolved tiles are
bisected and checked again.  Hitting any resource limit fails closed.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import argparse
import io
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter


PROCESS_STARTED = perf_counter()

import numpy as np
import torch
from torch import nn

from frozen_model import load_candidate as load_frozen_candidate
from systems import SYSTEMS
from verify_planar import (
    DEFAULT_MANIFEST,
    analytic_local_bound,
    load_candidate as load_proof_candidate,
)


HERE = Path(__file__).resolve().parent


def release_path(path):
    """Return a repository-relative path whenever possible."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(HERE).as_posix()
    except ValueError:
        return str(resolved)


def git_revision(module):
    """Best-effort source revision for the imported verifier package."""

    try:
        completed = subprocess.run(
            (
                "git",
                "-C",
                str(Path(module.__file__).resolve().parent),
                "rev-parse",
                "HEAD",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


@dataclass(frozen=True)
class TileFamily:
    """Axis-aligned boxes and the rule used to bisect unresolved boxes."""

    lower: torch.Tensor
    upper: torch.Tensor
    coordinate_scales: tuple[float, ...]
    logarithmic_coordinates: tuple[int, ...] = ()

    def __post_init__(self):
        if self.lower.ndim != 2 or self.lower.shape != self.upper.shape:
            raise ValueError("tile bounds must have shape (tiles, dimension)")
        if len(self.coordinate_scales) != self.lower.shape[1]:
            raise ValueError("one coordinate scale is required per dimension")
        widths = self.upper - self.lower
        if torch.any(widths < 0) or torch.any(torch.all(widths == 0, dim=1)):
            raise ValueError("every tile must have a nonempty positive-width axis")


class FrozenPlanarGraph(nn.Module):
    """Shared exact expressions for one frozen planar candidate."""

    def __init__(self, candidate):
        super().__init__()
        self.candidate = candidate
        origin = torch.zeros(
            (1, 2),
            dtype=candidate.input_scale.dtype,
            device=candidate.input_scale.device,
        )
        with torch.no_grad():
            correction_origin = candidate.correction_network(origin).reshape(())
        self.register_buffer("correction_origin", correction_origin)

    def core_terms(self, u1, u2):
        """Return log-shape and ``D_u log(shape)`` on the unit circle."""

        unit = torch.cat((u1, u2), dim=1)
        hidden = torch.tanh(
            torch.nn.functional.linear(
                unit,
                self.candidate.core_feature_weight,
                self.candidate.core_feature_bias,
            )
        )
        log_shape = torch.nn.functional.linear(
            hidden,
            self.candidate.core_output_weight.reshape(1, -1),
            self.candidate.core_output_bias.reshape(1),
        )
        angular_gradient = (
            self.candidate.core_output_weight * (1.0 - hidden.square())
        ) @ self.candidate.core_feature_weight
        return (
            log_shape,
            angular_gradient[:, :1],
            angular_gradient[:, 1:2],
        )

    def core_log_gradient_numerator(self, u1, u2):
        """Return ``q(u)`` such that ``D log(V_p)(r u) = q(u)/r``."""

        log_shape, angular1, angular2 = self.core_terms(u1, u2)
        angular_radial = angular1 * u1 + angular2 * u2
        q1 = (
            self.candidate.degree * u1
            + angular1
            - angular_radial * u1
        )
        q2 = (
            self.candidate.degree * u2
            + angular2
            - angular_radial * u2
        )
        return log_shape, q1, q2

    def correction_terms(self, x1, x2):
        """Return the anchored correction and its exact spatial gradient."""

        states = torch.cat((x1, x2), dim=1)
        scale = self.candidate.input_scale.reshape(-1)[0]
        inverse_scale = torch.reciprocal(scale)
        normalized = states * inverse_scale

        first = torch.tanh(
            torch.nn.functional.linear(
                normalized,
                self.candidate.correction_weights[0],
                self.candidate.correction_biases[0],
            )
        )
        first_slope = 1.0 - first.square()
        first_dx1 = (
            first_slope
            * self.candidate.correction_weights[0][:, 0]
            * inverse_scale
        )
        first_dx2 = (
            first_slope
            * self.candidate.correction_weights[0][:, 1]
            * inverse_scale
        )

        second = torch.tanh(
            torch.nn.functional.linear(
                first,
                self.candidate.correction_weights[1],
                self.candidate.correction_biases[1],
            )
        )
        second_slope = 1.0 - second.square()
        second_dx1 = second_slope * torch.nn.functional.linear(
            first_dx1, self.candidate.correction_weights[1]
        )
        second_dx2 = second_slope * torch.nn.functional.linear(
            first_dx2, self.candidate.correction_weights[1]
        )

        network_value = torch.nn.functional.linear(
            second, self.candidate.correction_output_weight
        )
        network_dx1 = torch.nn.functional.linear(
            second_dx1, self.candidate.correction_output_weight
        )
        network_dx2 = torch.nn.functional.linear(
            second_dx2, self.candidate.correction_output_weight
        )
        gain = self.candidate.correction_gain
        correction = gain * (network_value - self.correction_origin)
        return correction, gain * network_dx1, gain * network_dx2

    def latent_and_scaled_decrease(self, states):
        """Return ``(H, Phi)`` at nonzero Cartesian states."""

        x1, x2 = states[:, :1], states[:, 1:2]
        radius = (x1.square() + x2.square()).sqrt()
        inverse_radius = torch.reciprocal(radius)
        u1, u2 = x1 * inverse_radius, x2 * inverse_radius

        log_shape, q1, q2 = self.core_log_gradient_numerator(u1, u2)
        correction, correction_dx1, correction_dx2 = self.correction_terms(
            x1, x2
        )
        latent = radius.square() * torch.exp(log_shape + correction)

        if self.candidate.name == "two_machine":
            # f(x)/r for the degree-one dominant dynamics.
            scaled_f1 = u2
            scaled_f2 = (
                -0.5 * x2
                - torch.sin(x1 + math.pi / 3.0)
                + math.sin(math.pi / 3.0)
            ) * inverse_radius
        elif self.candidate.name == "cubic":
            # f(x)/r^3, written without a removable singularity.
            scaled_f1 = (
                4.0 * u1**3
                - u1.square() * u2
                - 6.0 * u1 * u2.square()
                - u2**3
            )
            scaled_f2 = (
                u1**3
                + 4.0 * u1.square() * u2
                + u1 * u2.square()
                - 6.0 * u2**3
            )
            degree_five = 0.2 * radius.square() * (
                3.0 + torch.sin(x1) + torch.cos(x2)
            )
            degree_seven = 0.05 * radius**4 * (
                3.0
                + torch.cos(x1 + x2)
                + torch.sin(x1 - x2)
            )
            modulation = degree_five + degree_seven
            scaled_f1 = scaled_f1 + modulation * u1
            scaled_f2 = scaled_f2 + modulation * u2
        else:  # pragma: no cover - candidate loading prevents this branch.
            raise ValueError(f"unsupported planar candidate {self.candidate.name}")

        scaled_gradient1 = q1 + radius * correction_dx1
        scaled_gradient2 = q2 + radius * correction_dx2
        phi = (
            scaled_gradient1 * scaled_f1
            + scaled_gradient2 * scaled_f2
        )
        return latent, phi

    def latent_value(self, states):
        """Return only ``H``; this keeps boundary graphs minimal."""

        x1, x2 = states[:, :1], states[:, 1:2]
        radius = (x1.square() + x2.square()).sqrt()
        inverse_radius = torch.reciprocal(radius)
        u1, u2 = x1 * inverse_radius, x2 * inverse_radius
        log_shape, _, _ = self.core_terms(u1, u2)
        correction, _, _ = self.correction_terms(x1, x2)
        return radius.square() * torch.exp(log_shape + correction)


class CoreCircleGraph(FrozenPlanarGraph):
    """Homogeneous logarithmic derivative parameterized by angle."""

    def forward(self, angle):
        u1, u2 = torch.cos(angle), torch.sin(angle)
        _, q1, q2 = self.core_log_gradient_numerator(u1, u2)
        if self.candidate.name == "two_machine":
            f1 = u2
            f2 = -0.5 * u1 - 0.5 * u2
        else:
            f1 = (
                4.0 * u1**3
                - u1.square() * u2
                - 6.0 * u1 * u2.square()
                - u2**3
            )
            f2 = (
                u1**3
                + 4.0 * u1.square() * u2
                + u1 * u2.square()
                - 6.0 * u2**3
            )
        return q1 * f1 + q2 * f2


class OuterDecreaseGraph(FrozenPlanarGraph):
    """Outputs ``(H, Phi)`` on Cartesian tiles away from the origin."""

    def forward(self, states):
        latent, phi = self.latent_and_scaled_decrease(states)
        return torch.cat((latent, phi), dim=1)


class BoundaryLevelGraph(FrozenPlanarGraph):
    """Latent value on Cartesian inputs; one input interval is degenerate."""

    def forward(self, states):
        return self.latent_value(states)


def _sync_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _split_tiles(family, unresolved):
    """Bisect every unresolved tile along its least-resolved coordinate."""

    children_lower = []
    children_upper = []
    logarithmic = set(family.logarithmic_coordinates)
    for lower, upper in zip(
        family.lower[unresolved].cpu(), family.upper[unresolved].cpu()
    ):
        scores = []
        for index, scale in enumerate(family.coordinate_scales):
            lo = float(lower[index])
            hi = float(upper[index])
            if index in logarithmic:
                if not 0.0 < lo < hi:
                    raise ValueError("logarithmic tile coordinates must be positive")
                score = math.log(hi / lo) / scale
            else:
                score = (hi - lo) / scale
            scores.append(score)
        coordinate = int(np.argmax(scores))
        lo = float(lower[coordinate])
        hi = float(upper[coordinate])
        if coordinate in logarithmic:
            midpoint = math.sqrt(lo * hi)
        else:
            midpoint = lo + (hi - lo) / 2.0
        if not lo < midpoint < hi:
            raise RuntimeError("floating-point subdivision made no progress")

        first_upper = upper.clone()
        first_upper[coordinate] = midpoint
        second_lower = lower.clone()
        second_lower[coordinate] = midpoint
        children_lower.extend((lower.clone(), second_lower))
        children_upper.extend((first_upper, upper.clone()))

    return TileFamily(
        lower=torch.stack(children_lower),
        upper=torch.stack(children_upper),
        coordinate_scales=family.coordinate_scales,
        logarithmic_coordinates=family.logarithmic_coordinates,
    )


def _crown_round(
    abcrown,
    graph,
    family,
    clause_matrix,
    clause_rhs,
    *,
    device,
    batch_size,
    round_timeout,
    verbose,
):
    """Run one batched CROWN pass and return unresolved tile indices."""

    specification = abcrown.VerificationSpec.build_spec(
        lower=family.lower,
        upper=family.upper,
        clauses=[(clause_matrix, clause_rhs)],
    )
    config = abcrown.ConfigBuilder.from_defaults().set(
        general__device=device.type,
        general__double_fp=False,
        general__complete_verifier="skip",
        solver__bound_prop_method="crown",
        solver__batch_size=batch_size,
        solver__optimize_disjuncts_separately=True,
        attack__pgd_order="skip",
        bab__timeout=round_timeout,
    )
    solver = abcrown.ABCrownSolver(
        specification,
        graph,
        config=config,
    )

    captured = io.StringIO()
    stdout_context = redirect_stdout(captured) if not verbose else nullcontext()
    stderr_context = redirect_stderr(captured) if not verbose else nullcontext()
    _sync_cuda(device)
    started = perf_counter()
    try:
        with stdout_context, stderr_context:
            result = solver.solve(return_reference=False)
    except Exception as error:
        _sync_cuda(device)
        return {
            "elapsed_seconds": perf_counter() - started,
            "status": "error",
            "diagnostic": f"{type(error).__name__}: {error}",
            "backend_log_tail": captured.getvalue()[-4000:],
            "unresolved": list(range(len(family.lower))),
        }
    _sync_cuda(device)
    elapsed = perf_counter() - started
    status = str(result.status)
    if bool(result.success) and status.startswith("safe"):
        unresolved = []
    else:
        handler = solver.spec_handler_incomplete
        if handler is None or not hasattr(handler, "unverified_or_indices"):
            return {
                "elapsed_seconds": elapsed,
                "status": status,
                "diagnostic": "backend did not identify unresolved tiles",
                "backend_log_tail": captured.getvalue()[-4000:],
                "unresolved": list(range(len(family.lower))),
            }
        unresolved = handler.unverified_or_indices.detach().cpu().tolist()
    return {
        "elapsed_seconds": elapsed,
        "status": status,
        "unresolved": unresolved,
    }


class nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        return False


def adaptive_verify(
    abcrown,
    name,
    graph,
    initial_family,
    clause_matrix,
    clause_rhs,
    *,
    device,
    batch_size,
    round_timeout,
    maximum_rounds,
    maximum_tiles,
    verbose,
):
    """Exhaustively verify a tiled property, refining only unknown tiles."""

    family = initial_family
    terminal_tiles = 0
    total_tiles_checked = 0
    rounds = []
    started = perf_counter()
    for round_index in range(1, maximum_rounds + 1):
        tile_count = len(family.lower)
        total_tiles_checked += tile_count
        outcome = _crown_round(
            abcrown,
            graph,
            family,
            clause_matrix,
            clause_rhs,
            device=device,
            batch_size=batch_size,
            round_timeout=round_timeout,
            verbose=verbose,
        )
        unresolved = outcome.pop("unresolved")
        verified_now = tile_count - len(unresolved)
        terminal_tiles += verified_now
        rounds.append(
            {
                "round": round_index,
                "tiles": tile_count,
                "verified": verified_now,
                "unresolved": len(unresolved),
                **outcome,
            }
        )
        if verbose:
            print(
                f"{name}: round {round_index}, {verified_now}/{tile_count} "
                "tiles verified"
            )
        if outcome["status"] == "error":
            return {
                "name": name,
                "verified": False,
                "elapsed_seconds": perf_counter() - started,
                "terminal_verified_tiles": terminal_tiles,
                "total_tiles_checked": total_tiles_checked,
                "remaining_tiles": len(unresolved),
                "rounds": rounds,
                "failure": "CROWN backend error",
            }
        if not unresolved:
            return {
                "name": name,
                "verified": True,
                "elapsed_seconds": perf_counter() - started,
                "terminal_verified_tiles": terminal_tiles,
                "total_tiles_checked": total_tiles_checked,
                "remaining_tiles": 0,
                "rounds": rounds,
            }
        if round_index == maximum_rounds:
            break
        next_family = _split_tiles(family, unresolved)
        if len(next_family.lower) > maximum_tiles:
            return {
                "name": name,
                "verified": False,
                "elapsed_seconds": perf_counter() - started,
                "terminal_verified_tiles": terminal_tiles,
                "total_tiles_checked": total_tiles_checked,
                "remaining_tiles": len(next_family.lower),
                "rounds": rounds,
                "failure": "maximum tile count exceeded",
            }
        family = next_family

    return {
        "name": name,
        "verified": False,
        "elapsed_seconds": perf_counter() - started,
        "terminal_verified_tiles": terminal_tiles,
        "total_tiles_checked": total_tiles_checked,
        "remaining_tiles": len(unresolved),
        "rounds": rounds,
        "failure": "maximum subdivision rounds exhausted",
    }


def linear_tiles(lower, upper, count):
    edges = torch.linspace(float(lower), float(upper), count + 1)
    return TileFamily(
        lower=edges[:-1].reshape(-1, 1),
        upper=edges[1:].reshape(-1, 1),
        coordinate_scales=(float(upper) - float(lower),),
    )


def complement_rectangles(domain, inner_half_width):
    """Return four rectangles covering the complement of an inner square."""

    (x1_lower, x1_upper), (x2_lower, x2_upper) = domain
    delta = float(inner_half_width)
    rectangles = (
        ((x1_lower, -delta), (x2_lower, x2_upper)),
        ((delta, x1_upper), (x2_lower, x2_upper)),
        ((-delta, delta), (x2_lower, -delta)),
        ((-delta, delta), (delta, x2_upper)),
    )
    lower = torch.tensor([[box[0][0], box[1][0]] for box in rectangles])
    upper = torch.tensor([[box[0][1], box[1][1]] for box in rectangles])
    return TileFamily(
        lower=lower,
        upper=upper,
        coordinate_scales=(
            float(x1_upper) - float(x1_lower),
            float(x2_upper) - float(x2_lower),
        ),
    )


def boundary_face_tiles(domain, fixed_coordinate, fixed_value, count):
    """Tile a box face with an exact zero-width fixed coordinate."""

    free_coordinate = 1 - int(fixed_coordinate)
    edges = torch.linspace(
        float(domain[free_coordinate][0]),
        float(domain[free_coordinate][1]),
        count + 1,
    )
    lower = torch.empty((count, 2))
    upper = torch.empty((count, 2))
    lower[:, free_coordinate] = edges[:-1]
    upper[:, free_coordinate] = edges[1:]
    lower[:, fixed_coordinate] = float(fixed_value)
    upper[:, fixed_coordinate] = float(fixed_value)
    return TileFamily(
        lower=lower,
        upper=upper,
        coordinate_scales=(
            float(domain[0][1]) - float(domain[0][0]),
            float(domain[1][1]) - float(domain[1][0]),
        ),
    )


def implementation_check(candidate, candidate_name):
    """Numerically check the handwritten graph against the public evaluator."""

    domain = tuple(tuple(map(float, pair)) for pair in candidate.metadata["domain"])
    generator = torch.Generator(device="cpu").manual_seed(20260816)
    unit = torch.rand((4096, 2), generator=generator)
    lower = torch.tensor([pair[0] for pair in domain])
    upper = torch.tensor([pair[1] for pair in domain])
    states = lower + unit * (upper - lower)
    keep = torch.linalg.vector_norm(states, dim=1) > 1e-3
    states = states[keep]

    graph = OuterDecreaseGraph(candidate)
    with torch.no_grad():
        actual = graph(states)
        expected_latent = candidate.latent(states)
    differentiable = states.detach().requires_grad_(False)
    dynamics = SYSTEMS[candidate_name](differentiable)
    gradient = candidate.log_gradient(differentiable)
    radius = torch.linalg.vector_norm(differentiable, dim=1, keepdim=True)
    degree = int(candidate.metadata["dominant_degree"])
    expected_phi = radius ** (1 - degree) * torch.sum(
        gradient * dynamics, dim=1, keepdim=True
    )
    latent_error = float((actual[:, :1] - expected_latent).abs().max())
    latent_scaled_error = float(
        (
            (actual[:, :1] - expected_latent).abs()
            / (1.0 + expected_latent.abs())
        ).max()
    )
    phi_error = float((actual[:, 1:2] - expected_phi).abs().max())
    latent_tolerance = 2e-5
    phi_tolerance = 5e-5
    if latent_scaled_error > latent_tolerance or phi_error > phi_tolerance:
        raise RuntimeError(
            "handwritten CROWN graph disagrees with frozen evaluator: "
            f"scaled latent error={latent_scaled_error}, Phi error={phi_error}"
        )
    return {
        "points": len(states),
        "maximum_absolute_latent_error": latent_error,
        "maximum_scaled_latent_error": latent_scaled_error,
        "maximum_absolute_scaled_decrease_error": phi_error,
        "scaled_latent_tolerance": latent_tolerance,
        "scaled_decrease_tolerance": phi_tolerance,
        "passed": True,
    }


def verify_candidate(args):
    try:
        import abcrown
    except ImportError as error:
        raise RuntimeError(
            "Install alpha-beta-CROWN so that `import abcrown` succeeds."
        ) from error

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this release script requires --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    manifest = Path(args.manifest).resolve()
    proof_candidate = load_proof_candidate(args.candidate, manifest)
    candidate = load_frozen_candidate(args.candidate, manifest).float().eval()
    local = analytic_local_bound(proof_candidate)
    if not local["passed"]:
        raise RuntimeError("the retained analytic local bound does not close")
    graph_check = implementation_check(candidate, args.candidate)

    domain = tuple(tuple(map(float, pair)) for pair in candidate.metadata["domain"])
    proof_margin = float(args.proof_margin)
    eta = float(candidate.metadata["core_margin"])
    outer_margin = float(candidate.metadata["outer_margin"])
    latent_level = float(candidate.metadata["latent_level"])

    verification_started = perf_counter()
    angle_limit = float(np.float32(math.nextafter(math.pi, math.inf)))
    core_family = linear_tiles(-angle_limit, angle_limit, args.core_tiles)
    core = adaptive_verify(
        abcrown,
        "homogeneous_core_unit_circle",
        CoreCircleGraph(candidate),
        core_family,
        -torch.ones((1, 1)),
        torch.tensor((eta + proof_margin,)),
        device=device,
        batch_size=args.batch_size,
        round_timeout=args.round_timeout,
        maximum_rounds=args.maximum_rounds,
        maximum_tiles=args.maximum_tiles,
        verbose=args.verbose,
    )

    obligations = [core]
    if core["verified"]:
        rho = float(candidate.metadata["local_radius"])
        inner_half_width = float(np.float32(rho / 2.0))
        if math.sqrt(2.0) * inner_half_width >= rho:
            raise RuntimeError("the inner square is not inside the local ball")
        outer_family = complement_rectangles(domain, inner_half_width)
        outer = adaptive_verify(
            abcrown,
            "outer_sublevel_decrease",
            OuterDecreaseGraph(candidate),
            outer_family,
            torch.tensor(((1.0, 0.0), (0.0, -1.0))),
            torch.tensor(
                (
                    latent_level + proof_margin,
                    outer_margin + proof_margin,
                )
            ),
            device=device,
            batch_size=args.batch_size,
            round_timeout=args.round_timeout,
            maximum_rounds=args.maximum_rounds,
            maximum_tiles=args.maximum_tiles,
            verbose=args.verbose,
        )
        outer["inner_square_half_width"] = inner_half_width
        outer["inner_square_corner_radius"] = math.sqrt(2.0) * inner_half_width
        outer["local_ball_radius"] = rho
        obligations.append(outer)

    if all(item["verified"] for item in obligations):
        labels = ("x1_lower", "x1_upper", "x2_lower", "x2_upper")
        sides = (
            (0, domain[0][0], domain[1]),
            (0, domain[0][1], domain[1]),
            (1, domain[1][0], domain[0]),
            (1, domain[1][1], domain[0]),
        )
        for label, (coordinate, value, free_bounds) in zip(labels, sides):
            face = adaptive_verify(
                abcrown,
                f"boundary_{label}",
                BoundaryLevelGraph(candidate),
                boundary_face_tiles(
                    domain, coordinate, value, args.boundary_tiles
                ),
                torch.ones((1, 1)),
                torch.tensor((latent_level + proof_margin,)),
                device=device,
                batch_size=args.batch_size,
                round_timeout=args.round_timeout,
                maximum_rounds=args.maximum_rounds,
                maximum_tiles=args.maximum_tiles,
                verbose=args.verbose,
            )
            obligations.append(face)
            if not face["verified"]:
                break

    verified = len(obligations) == 6 and all(
        item["verified"] for item in obligations
    )
    _sync_cuda(device)
    verification_elapsed = perf_counter() - verification_started
    payload = {
        "schema": "standalone-planar-gpu-crown-proof-v1",
        "candidate": args.candidate,
        "verified": verified,
        "claim": (
            "strict Lyapunov sublevel on the complete verification box"
            if verified
            else "GPU CROWN verification failed or was inconclusive"
        ),
        "candidate_artifact": {
            "manifest": release_path(manifest),
            "portable_weights": release_path(proof_candidate.weights_path),
            "portable_weights_sha256": proof_candidate.weights_sha256,
            "correction_gain": candidate.correction_gain,
            "latent_level": latent_level,
            "bounded_level": float(candidate.metadata["bounded_level"]),
            "domain": domain,
        },
        "method": {
            "software": "alpha-beta-CROWN",
            "software_version": str(getattr(abcrown, "__version__", "unknown")),
            "software_commit": git_revision(abcrown),
            "bound_method": "CROWN",
            "complete_verifier": "exhaustive adaptive input subdivision",
            "dtype": "float32",
            "proof_margin": proof_margin,
            "stopping_criterion": (
                "every terminal tile has a certified CROWN lower bound above "
                "the corresponding strengthened specification threshold"
            ),
            "failure_policy": "fail closed on any unresolved tile or backend error",
            "settings": {
                "core_initial_tiles": args.core_tiles,
                "boundary_initial_tiles_per_face": args.boundary_tiles,
                "batch_size": args.batch_size,
                "round_timeout_seconds": args.round_timeout,
                "maximum_subdivision_rounds": args.maximum_rounds,
                "maximum_tiles_per_round": args.maximum_tiles,
            },
            "proved_properties": {
                "core": "q_p(u)^T f_d(u) < -(eta + proof_margin) for every ||u||=1",
                "local": "the analytic beta bound satisfies beta < eta on ||x|| <= rho",
                "outer": (
                    "H(x) <= c implies Phi(x) < -(outer_margin + proof_margin) "
                    "on D outside the inner square"
                ),
                "boundary": "H(x) > c + proof_margin on every face of D",
                "coverage": (
                    "the inner square is contained in B_rho; its four-rectangle "
                    "complement and the inner square cover the complete box D"
                ),
            },
        },
        "hardware": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
        "implementation_check": graph_check,
        "analytic_local_bound": local,
        "obligations": obligations,
        "verification_elapsed_seconds": verification_elapsed,
        "process_elapsed_seconds": perf_counter() - PROCESS_STARTED,
    }
    return payload


def write_payload(payload, output, quiet=False):
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    if not quiet:
        print(text, end="")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify a frozen planar candidate with GPU CROWN bounds."
    )
    parser.add_argument("candidate", choices=("two_machine", "cubic"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--proof-margin", type=float, default=1e-6)
    parser.add_argument("--core-tiles", type=int, default=512)
    parser.add_argument("--boundary-tiles", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--round-timeout", type=float, default=600.0)
    parser.add_argument("--maximum-rounds", type=int, default=56)
    parser.add_argument("--maximum-tiles", type=int, default=1_000_000)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ("proof_margin", "round_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in (
        "core_tiles",
        "boundary_tiles",
        "batch_size",
        "maximum_rounds",
        "maximum_tiles",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main():
    args = parse_args()
    payload = verify_candidate(args)
    write_payload(payload, args.output, quiet=args.quiet)
    return 0 if payload["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
