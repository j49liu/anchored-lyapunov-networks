#!/usr/bin/env python3
"""Replay the two planar formal certificates from portable frozen weights.

This file deliberately contains the complete verification model.  It imports
neither LyZNet nor any training code.  The only inputs are ``candidates.json``
and the candidate's NumPy ``.npz`` file.

For ``x = r u``, ``||u|| = 1``, define

    Phi(r, u) = r**(1-d) D log(H)(r u) f(r u),

where ``d`` is the dominant degree.  The proof has three overlapping pieces:

1. dReal excludes a violation of ``Phi(0, u) < -eta`` on the unit circle.
2. An analytic bound ``|Phi(r,u)-Phi(0,u)| <= beta < eta`` covers
   ``0 < r <= rho``.
3. dReal excludes a decrease violation on ``rho <= r`` inside the frozen
   sublevel, and excludes that sublevel from all four faces of the box.

Every solver obligation is a counterexample query.  A certificate is accepted
only if the analytic inequality closes and all six queries return ``UNSAT``.
Any exception, ``delta-sat`` result, missing query, or malformed artifact is a
failure rather than an inconclusive success.

The paper's system constants are encoded exactly: the swing query uses the
symbolic expression ``sqrt(3)``, and the cubic query clears the denominators
5 and 20 by multiplying the scaled dynamics by 20.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "artifacts" / "candidates.json"
PRECISION = 1e-4


@dataclass(frozen=True)
class Candidate:
    """Validated metadata and exact float32 parameter arrays."""

    name: str
    metadata: dict
    arrays: dict[str, np.ndarray]
    manifest_path: Path
    weights_path: Path
    weights_sha256: str

    @property
    def domain(self) -> tuple[tuple[float, float], ...]:
        return tuple(tuple(map(float, pair)) for pair in self.metadata["domain"])

    @property
    def degree(self) -> int:
        return int(self.metadata["dominant_degree"])

    @property
    def core_degree(self) -> int:
        return int(self.metadata["homogeneous_degree"])

    @property
    def local_radius(self) -> float:
        return float(self.metadata["local_radius"])

    @property
    def core_margin(self) -> float:
        return float(self.metadata["core_margin"])

    @property
    def outer_margin(self) -> float:
        return float(self.metadata["outer_margin"])

    @property
    def correction_gain(self) -> float:
        return float(self.metadata["correction_gain"])


@dataclass(frozen=True)
class Query:
    """One dReal counterexample obligation."""

    name: str
    formula: object


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_path(path: Path) -> str:
    """Prefer a portable path when an artifact is inside this release."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(HERE).as_posix()
    except ValueError:
        return str(resolved)


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def load_candidate(name: str, manifest_path: Path) -> Candidate:
    """Load one candidate and reject every metadata/array inconsistency."""

    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "frozen-anchored-lyapunov-candidates-v1":
        raise ValueError("unsupported candidate-manifest schema")
    if name not in manifest.get("candidates", {}):
        raise ValueError(f"candidate {name!r} is absent from the manifest")
    metadata = manifest["candidates"][name]

    expected = {
        "two_machine": {
            "domain": ((-2.0, 3.0), (-3.0, 1.5)),
            "dominant_degree": 1,
        },
        "cubic": {
            "domain": ((-1.5, 1.5), (-1.5, 1.5)),
            "dominant_degree": 3,
        },
    }[name]
    domain = tuple(tuple(map(float, pair)) for pair in metadata.get("domain", ()))
    if domain != expected["domain"]:
        raise ValueError(f"unexpected verification domain for {name}")
    if int(metadata.get("dimension", -1)) != 2:
        raise ValueError("the planar verifier requires dimension two")
    if int(metadata.get("dominant_degree", -1)) != expected["dominant_degree"]:
        raise ValueError(f"unexpected dominant degree for {name}")
    if int(metadata.get("homogeneous_degree", -1)) != 2:
        raise ValueError("the frozen planar core must have degree two")
    if metadata.get("core_kind") != "homogeneous_tanh":
        raise ValueError("the frozen planar core must be homogeneous_tanh")
    if int(metadata.get("correction_depth", -1)) != 2:
        raise ValueError("the frozen planar correction must have two hidden layers")
    if not all(lower < 0.0 < upper for lower, upper in domain):
        raise ValueError("the origin must lie strictly inside the verification box")

    weights_path = (manifest_path.parent / metadata["portable_weights"]).resolve()
    actual_weights_hash = sha256(weights_path)
    if actual_weights_hash != metadata.get("portable_weights_sha256"):
        raise ValueError("portable-weight SHA-256 mismatch")
    with np.load(weights_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}

    records = metadata.get("array_records", {})
    if set(arrays) != set(records):
        raise ValueError("portable arrays differ from the manifest inventory")
    for key, array in arrays.items():
        record = records[key]
        if list(array.shape) != record.get("shape"):
            raise ValueError(f"shape mismatch for {key}")
        if str(array.dtype) != record.get("dtype"):
            raise ValueError(f"dtype mismatch for {key}")
        if array_sha256(array) != record.get("raw_bytes_sha256"):
            raise ValueError(f"raw-array SHA-256 mismatch for {key}")
        if not np.isfinite(array).all():
            raise ValueError(f"nonfinite value in {key}")

    required_shapes = {
        "core_feature_weight": (48, 2),
        "core_feature_bias": (48,),
        "core_output_weight": (48,),
        "core_output_bias": (),
        "input_scale": (2,),
        "correction_weight_0": (32, 2),
        "correction_bias_0": (32,),
        "correction_weight_1": (32, 32),
        "correction_bias_1": (32,),
        "correction_output_weight": (1, 32),
    }
    if {key: tuple(value.shape) for key, value in arrays.items()} != required_shapes:
        raise ValueError("the portable arrays do not encode the frozen 1x48/2x32 model")

    input_scale = arrays["input_scale"].reshape(-1)
    if not np.all(input_scale == input_scale[0]) or not input_scale[0] > 0:
        raise ValueError("formal replay requires one positive scalar input scale")
    latent_level = float(metadata.get("latent_level", math.nan))
    bounded_level = float(metadata.get("bounded_level", math.nan))
    if not (math.isfinite(latent_level) and latent_level > 0):
        raise ValueError("the latent level must be positive and finite")
    if not math.isclose(
        math.tanh(latent_level), bounded_level, rel_tol=0.0, abs_tol=2e-15
    ):
        raise ValueError("bounded_level is not tanh(latent_level)")
    if not (0.0 < bounded_level < 1.0):
        raise ValueError("the bounded level must lie strictly between zero and one")
    for key in ("local_radius", "core_margin", "outer_margin"):
        value = float(metadata.get(key, math.nan))
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{key} must be positive and finite")
    if float(metadata["local_radius"]) >= min(
        min(-lower, upper) for lower, upper in domain
    ):
        raise ValueError("the local ball is not strictly inside the box")

    return Candidate(
        name=name,
        metadata=metadata,
        arrays=arrays,
        manifest_path=manifest_path,
        weights_path=weights_path,
        weights_sha256=actual_weights_hash,
    )


def _upper_float(value: Fraction) -> float:
    """Smallest nearby float known to be no smaller than ``value``."""

    candidate = float(value)
    while Fraction.from_float(candidate) < value:
        candidate = math.nextafter(candidate, math.inf)
    return candidate


def _upper_sqrt(value: Fraction) -> float:
    """Outward-rounded square root of a nonnegative rational."""

    if value < 0:
        raise ValueError("cannot bound the square root of a negative number")
    candidate = math.sqrt(float(value))
    while Fraction.from_float(candidate) ** 2 < value:
        candidate = math.nextafter(candidate, math.inf)
    return candidate


def _gershgorin_gram_bound(matrix: list[list[Fraction]]) -> float:
    rows = len(matrix)
    columns = len(matrix[0])
    gram = [
        [
            sum(
                (matrix[row][first] * matrix[row][second] for row in range(rows)),
                Fraction(0),
            )
            for second in range(columns)
        ]
        for first in range(columns)
    ]
    eigenvalue_bound = max(
        gram[index][index]
        + sum(
            (
                abs(gram[index][other])
                for other in range(columns)
                if other != index
            ),
            Fraction(0),
        )
        for index in range(columns)
    )
    return _upper_sqrt(eigenvalue_bound)


def matrix_norm_upper(array: np.ndarray) -> float:
    """Rigorous spectral-norm upper bound for a stored binary-float matrix."""

    matrix = [
        [Fraction.from_float(float(value)) for value in row]
        for row in np.asarray(array).reshape(array.shape[0], -1)
    ]
    transpose = [list(row) for row in zip(*matrix)]
    # A and A.T have the same nonzero singular values.  Either Gershgorin
    # bound is valid, hence their minimum remains a valid upper bound.
    return min(
        _gershgorin_gram_bound(matrix),
        _gershgorin_gram_bound(transpose),
    )


def spectral_product(*arrays: np.ndarray) -> float:
    product = Fraction(1)
    for array in arrays:
        product *= Fraction.from_float(matrix_norm_upper(array))
    return _upper_float(product)


def analytic_local_bound(candidate: Candidate) -> dict:
    """Compute the explicit local perturbation bound ``beta``.

    Tanh is 1-Lipschitz.  Therefore, on the unit circle,

        ||D log(V_p)|| <= p + ||a|| ||A|| = M_V,
        ||D G|| <= gain ||C|| ||B_2|| ||B_1|| / scale = M_G.

    If ``||f(ru)/r^d-f_d(u)|| <= epsilon(rho)`` and
    ``||f_d(u)|| <= M_f``, the polar identity gives

        beta = M_V epsilon(rho) + rho M_G (M_f + epsilon(rho)).

    All matrix-norm arithmetic is outward rounded from the exact binary
    floats stored in the NPZ file.
    """

    arrays = candidate.arrays
    core_angular = spectral_product(
        arrays["core_feature_weight"],
        arrays["core_output_weight"].reshape(1, -1),
    )
    core_bound_fraction = (
        Fraction(candidate.core_degree)
        + Fraction.from_float(core_angular)
    )
    core_bound = _upper_float(core_bound_fraction)

    correction_network = spectral_product(
        arrays["correction_weight_0"],
        arrays["correction_weight_1"],
        arrays["correction_output_weight"],
    )
    input_scale = float(arrays["input_scale"].reshape(-1)[0])
    correction_bound_fraction = (
        Fraction.from_float(abs(candidate.correction_gain))
        * Fraction.from_float(correction_network)
        / Fraction.from_float(input_scale)
    )
    correction_bound = _upper_float(correction_bound_fraction)

    rho = Fraction.from_float(candidate.local_radius)
    if candidate.name == "two_machine":
        # Taylor's theorem applied to sin(pi/3+x1): the second derivative has
        # magnitude at most one.  After division by the degree-one r, the
        # scaled dynamics remainder is at most r/2.
        epsilon = rho / 2
        dominant_bound = _upper_sqrt(Fraction(2))
        remainder_description = "||E(r,u)|| <= r/2"
    else:
        # The complete noncubic part is
        #   [(1/5) r^2 A(r,u) + (1/20) r^4 B(r,u)] u
        # after division by r^3.  Both A and B lie in [1,5], so its norm is
        # at most r^2 (1 + r^2/4).
        epsilon = rho**2 * (1 + rho**2 / 4)
        dominant_bound = _upper_sqrt(Fraction(288))
        remainder_description = "||E(r,u)|| <= r^2*(1+r^2/4)"

    epsilon_upper = _upper_float(epsilon)
    beta_fraction = (
        Fraction.from_float(core_bound) * epsilon
        + rho
        * Fraction.from_float(correction_bound)
        * (Fraction.from_float(dominant_bound) + epsilon)
    )
    beta = _upper_float(beta_fraction)
    margin = candidate.core_margin
    passed = beta < margin
    return {
        "passed": passed,
        "local_radius": candidate.local_radius,
        "core_margin_eta": margin,
        "beta": beta,
        "strict_margin_eta_minus_beta": margin - beta,
        "core_log_gradient_bound": core_bound,
        "correction_gradient_bound": correction_bound,
        "dominant_vector_field_bound": dominant_bound,
        "scaled_dynamics_remainder_at_rho": epsilon_upper,
        "scaled_dynamics_remainder": remainder_description,
        "argument": (
            "beta=M_V*epsilon(rho)+rho*M_G*(M_f+epsilon(rho)); "
            "strict local decrease follows only when beta < eta"
        ),
    }


def affine_tanh_layer(dreal, values, weight, bias):
    """Write one frozen affine-tanh layer as a dReal expression list."""

    return [
        dreal.tanh(
            sum(float(coefficient) * value for coefficient, value in zip(row, values))
            + float(offset)
        )
        for row, offset in zip(weight, bias)
    ]


def correction_network(candidate: Candidate, dreal, states):
    arrays = candidate.arrays
    hidden = affine_tanh_layer(
        dreal,
        list(states),
        arrays["correction_weight_0"],
        arrays["correction_bias_0"],
    )
    hidden = affine_tanh_layer(
        dreal,
        hidden,
        arrays["correction_weight_1"],
        arrays["correction_bias_1"],
    )
    output = arrays["correction_output_weight"].reshape(-1)
    return sum(float(coefficient) * value for coefficient, value in zip(output, hidden))


def core_log_shape(candidate: Candidate, dreal, unit):
    arrays = candidate.arrays
    hidden = affine_tanh_layer(
        dreal,
        list(unit),
        arrays["core_feature_weight"],
        arrays["core_feature_bias"],
    )
    output = arrays["core_output_weight"].reshape(-1)
    return float(arrays["core_output_bias"].reshape(-1)[0]) + sum(
        float(coefficient) * value for coefficient, value in zip(output, hidden)
    )


def dominant_dynamics(candidate: Candidate, unit):
    u1, u2 = unit
    if candidate.name == "two_machine":
        return u2, -0.5 * u1 - 0.5 * u2
    return (
        4.0 * u1**3 - u1**2 * u2 - 6.0 * u1 * u2**2 - u2**3,
        u1**3 + 4.0 * u1**2 * u2 + u1 * u2**2 - 6.0 * u2**3,
    )


def scaled_full_dynamics(candidate: Candidate, dreal, radius, unit):
    """Return an integer multiple of ``f(r*u)/r^d`` and its multiplier."""

    u1, u2 = unit
    if candidate.name == "two_machine":
        angle = radius * u1
        flow = (
            u2,
            (
                -0.5 * radius * u2
                - 0.5 * dreal.sin(angle)
                - 0.5
                * dreal.sqrt(dreal.Expression(3.0))
                * (dreal.cos(angle) - 1.0)
            )
            / radius,
        )
        return flow, 1

    leading1, leading2 = dominant_dynamics(candidate, unit)
    angle1 = radius * u1
    angle2 = radius * u2
    degree_five_shape = 3.0 + dreal.sin(angle1) + dreal.cos(angle2)
    degree_seven_shape = (
        3.0
        + dreal.cos(angle1 + angle2)
        + dreal.sin(angle1 - angle2)
    )
    modulation_times_twenty = (
        4.0 * radius**2 * degree_five_shape
        + radius**4 * degree_seven_shape
    )
    flow_times_twenty = (
        20.0 * leading1 + modulation_times_twenty * u1,
        20.0 * leading2 + modulation_times_twenty * u2,
    )
    return flow_times_twenty, 20


def log_level_expression(candidate: Candidate, dreal):
    """Outward-enclose the frozen bounded level in solver arithmetic."""

    bounded = float(candidate.metadata["bounded_level"])
    enclosing = math.nextafter(bounded, 1.0)
    level = dreal.Expression(enclosing)
    one = dreal.Expression(1.0)
    atanh_level = dreal.Expression(0.5) * dreal.log((one + level) / (one - level))
    return dreal.log(atanh_level), enclosing


def maximum_box_radius(domain) -> float:
    squared = sum(
        max(
            Fraction.from_float(abs(float(lower))),
            Fraction.from_float(abs(float(upper))),
        )
        ** 2
        for lower, upper in domain
    )
    return _upper_sqrt(squared)


def build_queries(
    candidate: Candidate, dreal
) -> tuple[list[Query], float, int, float]:
    """Construct the six counterexample formulas used by the certificate."""

    # Query 1: the homogeneous core on the complete unit circle.
    core_unit = (
        dreal.Variable("core_u1"),
        dreal.Variable("core_u2"),
    )
    core_flow = dominant_dynamics(candidate, core_unit)
    core_radial_rate = sum(u * f for u, f in zip(core_unit, core_flow))
    core_tangent = tuple(
        f - u * core_radial_rate for u, f in zip(core_unit, core_flow)
    )
    shape = core_log_shape(candidate, dreal, core_unit)
    core_ratio = candidate.core_degree * core_radial_rate + sum(
        shape.Differentiate(u) * tangent
        for u, tangent in zip(core_unit, core_tangent)
    )
    unit_circle = sum(u**2 for u in core_unit) == 1
    core_bad = dreal.logical_and(
        unit_circle,
        core_ratio >= -candidate.core_margin,
    )

    # Queries 2--6: the exact frozen H on the outer polar shell.
    radius = dreal.Variable("strict_radius")
    unit = (
        dreal.Variable("strict_u1"),
        dreal.Variable("strict_u2"),
    )
    input_scale = float(candidate.arrays["input_scale"].reshape(-1)[0])
    normalized = tuple(radius * u / input_scale for u in unit)
    zero = (dreal.Expression(0.0), dreal.Expression(0.0))
    correction = candidate.correction_gain * (
        correction_network(candidate, dreal, normalized)
        - correction_network(candidate, dreal, zero)
    )
    full_shape = core_log_shape(candidate, dreal, unit) + correction
    log_latent = candidate.core_degree * dreal.log(radius) + full_shape
    scaled_flow, phi_multiplier = scaled_full_dynamics(
        candidate, dreal, radius, unit
    )
    radial_rate = sum(u * f for u, f in zip(unit, scaled_flow))
    tangent = tuple(f - u * radial_rate for u, f in zip(unit, scaled_flow))
    ratio = candidate.core_degree * radial_rate
    ratio += full_shape.Differentiate(radius) * radius * radial_rate
    ratio += sum(
        full_shape.Differentiate(u) * rate for u, rate in zip(unit, tangent)
    )

    sphere = sum(u**2 for u in unit) == 1
    radial = dreal.logical_and(
        radius >= candidate.local_radius,
        radius <= maximum_box_radius(candidate.domain),
    )
    box = dreal.logical_and(
        *(
            dreal.logical_and(radius * u >= lower, radius * u <= upper)
            for u, (lower, upper) in zip(unit, candidate.domain)
        )
    )
    shell = dreal.logical_and(sphere, radial, box)
    log_level, enclosing_level = log_level_expression(candidate, dreal)
    selected = log_latent <= log_level
    scaled_outer_margin = _upper_float(
        Fraction(phi_multiplier) * Fraction.from_float(candidate.outer_margin)
    )
    decrease_bad = ratio >= -scaled_outer_margin
    queries = [
        Query("homogeneous_core_unit_sphere", core_bad),
        Query(
            "outer_shell_counterexample",
            dreal.logical_and(shell, selected, decrease_bad),
        ),
    ]
    labels = ("x1_lower", "x1_upper", "x2_lower", "x2_upper")
    sides = (
        (0, candidate.domain[0][0]),
        (0, candidate.domain[0][1]),
        (1, candidate.domain[1][0]),
        (1, candidate.domain[1][1]),
    )
    for label, (coordinate, side) in zip(labels, sides):
        face = radius * unit[coordinate] == side
        queries.append(
            Query(
                f"boundary_{label}",
                dreal.logical_and(shell, face, selected),
            )
        )
    if len(queries) != 6:
        raise RuntimeError("internal error: expected exactly six queries")
    return queries, enclosing_level, phi_multiplier, scaled_outer_margin


def query_records(queries: list[Query], dump_directory: Path | None) -> list[dict]:
    records = []
    if dump_directory is not None:
        dump_directory.mkdir(parents=True, exist_ok=True)
    for index, query in enumerate(queries, start=1):
        text = str(query.formula)
        raw = (text + "\n").encode("utf-8")
        record = {
            "index": index,
            "name": query.name,
            "formula_characters": len(text),
            "formula_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "status": "not_run",
            "result": "not_run",
        }
        if dump_directory is not None:
            path = dump_directory / f"{index:02d}_{query.name}.txt"
            path.write_bytes(raw)
            record.update(
                {
                    "formula_path": str(path.resolve()),
                    "query_file_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        records.append(record)
    return records


def solver_config(dreal, jobs: int):
    config = dreal.Config()
    config.precision = PRECISION
    config.number_of_jobs = int(jobs)
    config.use_polytope_in_forall = True
    config.use_local_optimization = True
    return config


def run_queries(dreal, queries: list[Query], records: list[dict], jobs: int) -> bool:
    """Run in proof order and stop at the first non-UNSAT result."""

    config = solver_config(dreal, jobs)
    prerequisites_passed = True
    for query, record in zip(queries, records):
        if not prerequisites_passed:
            record["status"] = "not_run_prerequisite_failed"
            record["result"] = "not_run"
            continue
        record["status"] = "running"
        started = perf_counter()
        try:
            witness = dreal.CheckSatisfiability(query.formula, config)
        except Exception as error:  # Fail closed on solver/API errors.
            record.update(
                {
                    "status": "error",
                    "result": "error",
                    "elapsed_seconds": perf_counter() - started,
                    "diagnostic": f"{type(error).__name__}: {error}",
                }
            )
            prerequisites_passed = False
            continue
        record["elapsed_seconds"] = perf_counter() - started
        if witness is None:
            record.update({"status": "complete", "result": "unsat"})
        else:
            record.update(
                {
                    "status": "complete",
                    "result": "delta_sat",
                    "witness": str(witness),
                }
            )
            prerequisites_passed = False
    return len(records) == 6 and all(item.get("result") == "unsat" for item in records)


def write_result(payload: dict, output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a frozen planar Lyapunov certificate with dReal."
    )
    parser.add_argument("candidate", choices=("two_machine", "cubic"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="candidate manifest (default: artifacts/candidates.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and construct, but do not solve, all six queries",
    )
    parser.add_argument(
        "--dump-queries",
        type=Path,
        nargs="?",
        const=Path("queries"),
        help="write exact query text, optionally to the given directory",
    )
    parser.add_argument("--jobs", type=int, default=48)
    parser.add_argument("--output", type=Path, help="also write the JSON result")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")

    candidate = load_candidate(args.candidate, args.manifest)
    local = analytic_local_bound(candidate)
    if not local["passed"]:
        payload = {
            "schema": "standalone-planar-dreal-proof-v1",
            "candidate": candidate.name,
            "claim": "analytic local bound failed; no formal certificate",
            "dry_run": bool(args.dry_run),
            "verified": False,
            "local_bound": local,
            "queries": [],
        }
        write_result(payload, args.output)
        return 1

    try:
        import dreal
    except ImportError as error:
        raise RuntimeError(
            "dReal Python bindings are required to construct formal queries"
        ) from error

    queries, enclosing_level, phi_multiplier, scaled_outer_margin = build_queries(
        candidate, dreal
    )
    records = query_records(queries, args.dump_queries)
    started = perf_counter()
    all_unsat = False
    if not args.dry_run:
        all_unsat = run_queries(dreal, queries, records, args.jobs)
    elapsed = perf_counter() - started
    verified = bool(local["passed"] and all_unsat and not args.dry_run)
    payload = {
        "schema": "standalone-planar-dreal-proof-v1",
        "candidate": candidate.name,
        "claim": (
            "strict Lyapunov sublevel on the complete verification box"
            if verified
            else "dry run only; no solver claim"
            if args.dry_run
            else "formal verification failed or was inconclusive"
        ),
        "dry_run": bool(args.dry_run),
        "verified": verified,
        "all_required_queries_unsat": bool(all_unsat),
        "system_constants_encoded_exactly": True,
        "outer_query_phi_multiplier": phi_multiplier,
        "outer_query_scaled_margin": scaled_outer_margin,
        "candidate_artifact": {
            "manifest": release_path(candidate.manifest_path),
            "portable_weights": release_path(candidate.weights_path),
            "portable_weights_sha256": candidate.weights_sha256,
            "correction_gain": candidate.correction_gain,
            "latent_level": float(candidate.metadata["latent_level"]),
            "bounded_level": float(candidate.metadata["bounded_level"]),
            "formal_enclosing_bounded_level": enclosing_level,
            "domain": candidate.domain,
        },
        "local_bound": local,
        "solver": {
            "backend": "dreal",
            "version": str(getattr(dreal, "__version__", "unknown")),
            "precision": PRECISION,
            "jobs": args.jobs,
            "use_polytope_in_forall": True,
            "use_local_optimization": True,
            "elapsed_seconds": elapsed,
        },
        "queries": records,
    }
    write_result(payload, args.output)
    return 0 if args.dry_run or verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
