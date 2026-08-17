#!/usr/bin/env python3
"""Fail-closed integrity and portability checks for the public artifact."""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from frozen_model import candidate_names, load_candidate, load_manifest


HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "artifacts" / "candidates.json"
IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "queries",
    "runs",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mapped_pt_arrays(
    metadata: dict, state: dict[str, torch.Tensor]
) -> dict[str, np.ndarray]:
    """Map a frozen state dictionary according to the declared architecture."""

    core_kind = metadata["core_kind"]
    if core_kind == "quadratic":
        result = {"core_matrix": state["base.matrix"].cpu().numpy()}
    elif core_kind == "homogeneous_tanh":
        result = {
            "core_feature_weight": state["base.feature_weight"].cpu().numpy(),
            "core_feature_bias": state["base.feature_bias"].cpu().numpy(),
            "core_output_weight": state["base.output_weight"].cpu().numpy(),
            "core_output_bias": state["base.output_bias"].cpu().numpy(),
        }
    else:
        raise AssertionError(f"unsupported core kind: {core_kind}")

    depth = int(metadata["correction_depth"])
    result["input_scale"] = state["correction_net.input_scale"].cpu().numpy().reshape(-1)
    result["correction_output_weight"] = state[
        "correction_net.output.weight"
    ].cpu().numpy()
    for index in range(depth):
        result[f"correction_weight_{index}"] = state[
            f"correction_net.layers.{index}.weight"
        ].cpu().numpy()
        result[f"correction_bias_{index}"] = state[
            f"correction_net.layers.{index}.bias"
        ].cpu().numpy()
    return result


def evidence_records(metadata: dict):
    """Yield ``(role, path, sha256)`` for either supported manifest layout."""

    evidence = metadata.get("evidence", {})
    for role, record in evidence.items():
        if role.endswith("sha256"):
            continue
        if isinstance(record, dict):
            yield role, record["path"], record["sha256"]
        else:
            yield role, record, evidence.get(f"{role}_sha256")


def expected_pytorch_format(name: str, metadata: dict) -> str:
    """Return the public, weights-only container format for a candidate."""

    if name != "grune10d":
        return "discounted_canonical_anchored_state_dict_v1"
    if metadata["core_kind"] == "homogeneous_tanh":
        return "grune10d_homogeneous_anchored_state_dict_v1"
    return "grune10d_anchored_state_dict_v2"


def validate_candidate(name: str, metadata: dict):
    artifact_root = MANIFEST_PATH.parent
    npz_path = artifact_root / metadata["portable_weights"]
    pt_path = artifact_root / metadata["pytorch_weights"]
    assert sha256(npz_path) == metadata["portable_weights_sha256"]
    assert sha256(pt_path) == metadata["pytorch_weights_sha256"]
    if "frozen_pytorch_artifact_sha256" in metadata:
        assert metadata["frozen_pytorch_artifact_sha256"] == metadata[
            "pytorch_weights_sha256"
        ]
    assert math.isclose(
        math.tanh(metadata["latent_level"]),
        metadata["bounded_level"],
        rel_tol=0.0,
        abs_tol=2e-15,
    )

    with np.load(npz_path, allow_pickle=False) as portable:
        arrays = {key: portable[key] for key in portable.files}
    payload = torch.load(pt_path, map_location="cpu", weights_only=True)
    expected_format = expected_pytorch_format(name, metadata)
    assert payload["format"] == expected_format
    assert float(payload["selected_correction_scale"]) == float(
        metadata["correction_gain"]
    )
    assert float(payload["selected_latent_level"]) == float(metadata["latent_level"])
    assert float(payload["selected_bounded_level"]) == float(
        metadata["bounded_level"]
    )
    source = mapped_pt_arrays(metadata, payload["model_state_dict"])
    assert set(arrays) == set(source) == set(metadata["array_records"])
    for key, value in arrays.items():
        record = metadata["array_records"][key]
        contiguous = np.ascontiguousarray(value)
        assert list(value.shape) == record["shape"]
        assert str(value.dtype) == record["dtype"]
        assert hashlib.sha256(contiguous.tobytes(order="C")).hexdigest() == record[
            "raw_bytes_sha256"
        ]
        assert np.array_equal(value, source[key])
        assert np.isfinite(value).all()

    model = load_candidate(name).double()
    origin = torch.zeros((1, metadata["dimension"]), dtype=torch.float64)
    assert float(model.latent(origin)) == 0.0
    assert float(model.bounded(origin)) == 0.0

    for role, relative, expected in evidence_records(metadata):
        assert expected, f"missing evidence digest for {name}:{role}"
        assert sha256(artifact_root / relative) == expected

    if name != "grune10d":
        proof = json.loads(
            (
                artifact_root
                / metadata["evidence"]["historical_formal_proof"]
            ).read_text()
        )
        assert proof["verified"] and proof["all_required_queries_unsat"]
        assert len(proof["query_ledger"]) == 6
        assert all(item["result"] == "unsat" for item in proof["query_ledger"])
        assert proof["fixed_candidate"]["latent_level"] == metadata["latent_level"]
        assert proof["fixed_candidate"]["bounded_level"] == metadata["bounded_level"]

        replay = json.loads(
            (artifact_root / metadata["evidence"]["standalone_proof"]).read_text()
        )
        assert replay["schema"] == "standalone-planar-dreal-proof-v1"
        assert replay["candidate"] == name
        assert replay["verified"] and replay["all_required_queries_unsat"]
        assert replay["system_constants_encoded_exactly"]
        assert replay["outer_query_phi_multiplier"] == (
            1 if name == "two_machine" else 20
        )
        assert Fraction.from_float(replay["outer_query_scaled_margin"]) >= (
            Fraction(replay["outer_query_phi_multiplier"])
            * Fraction.from_float(metadata["outer_margin"])
        )
        assert replay["local_bound"]["passed"]
        assert replay["local_bound"]["beta"] == metadata[
            "retained_local_error_bound"
        ]
        assert replay["local_bound"]["beta"] < replay["local_bound"][
            "core_margin_eta"
        ]
        expected_queries = (
            "homogeneous_core_unit_sphere",
            "outer_shell_counterexample",
            "boundary_x1_lower",
            "boundary_x1_upper",
            "boundary_x2_lower",
            "boundary_x2_upper",
        )
        assert tuple(item["name"] for item in replay["queries"]) == expected_queries
        assert all(item["result"] == "unsat" for item in replay["queries"])
        assert all(len(item["formula_sha256"]) == 64 for item in replay["queries"])
        artifact = replay["candidate_artifact"]
        assert artifact["portable_weights_sha256"] == metadata[
            "portable_weights_sha256"
        ]
        assert artifact["latent_level"] == metadata["latent_level"]
        assert artifact["bounded_level"] == metadata["bounded_level"]
        assert artifact["formal_enclosing_bounded_level"] >= metadata[
            "bounded_level"
        ]
        assert not Path(artifact["manifest"]).is_absolute()
        assert not Path(artifact["portable_weights"]).is_absolute()
        assert replay["solver"]["version"] == "4.21.6.2"
        assert replay["solver"]["precision"] == 1e-4

        gpu_record = metadata["evidence"].get("gpu_crown_proof")
        if gpu_record is not None:
            if isinstance(gpu_record, dict):
                gpu_relative = gpu_record["path"]
            else:
                gpu_relative = gpu_record
            gpu = json.loads((artifact_root / gpu_relative).read_text())
            assert gpu["schema"] == "standalone-planar-gpu-crown-proof-v1"
            assert gpu["candidate"] == name
            assert gpu["verified"]
            assert gpu["implementation_check"]["passed"]
            assert gpu["analytic_local_bound"]["passed"]
            assert len(gpu["obligations"]) == 6
            assert all(item["verified"] for item in gpu["obligations"])
            assert gpu["method"]["software"] == "alpha-beta-CROWN"
            assert gpu["method"]["software_version"] == "0.7.0"
            assert gpu["method"]["software_commit"] == (
                "6b8bbcfac1c01da1cabd240a87e4dce1a65f5a2b"
            )
            assert gpu["method"]["bound_method"] == "CROWN"
            assert gpu["method"]["dtype"] == "float32"
            assert gpu["method"]["trust_scope"] == (
                "float32 backend without a proved roundoff envelope; "
                "corroborating evidence rather than a real-arithmetic formal certificate"
            )
            assert gpu["method"]["complete_verifier"] == (
                "exhaustive adaptive input subdivision"
            )
            assert gpu["method"]["proof_margin"] == 1e-6
            expected_gpu_obligations = (
                "homogeneous_core_unit_circle",
                "outer_sublevel_decrease",
                "boundary_x1_lower",
                "boundary_x1_upper",
                "boundary_x2_lower",
                "boundary_x2_upper",
            )
            assert tuple(item["name"] for item in gpu["obligations"]) == (
                expected_gpu_obligations
            )
            assert all(item["remaining_tiles"] == 0 for item in gpu["obligations"])
            gpu_artifact = gpu["candidate_artifact"]
            assert gpu_artifact["portable_weights_sha256"] == metadata[
                "portable_weights_sha256"
            ]
            assert gpu_artifact["latent_level"] == metadata["latent_level"]
            assert gpu_artifact["bounded_level"] == metadata["bounded_level"]
            assert not Path(gpu_artifact["manifest"]).is_absolute()
            assert not Path(gpu_artifact["portable_weights"]).is_absolute()


def released_files() -> tuple[Path, ...]:
    """Return the stable checksum inventory, excluding VCS/cache/run state."""

    return tuple(
        sorted(
            item
            for item in HERE.rglob("*")
            if item.is_file()
            and item.name != "SHA256SUMS"
            and item.suffix != ".pyc"
            and not IGNORED_DIRECTORIES.intersection(item.relative_to(HERE).parts)
        )
    )


def validate_sha256sums():
    path = HERE / "SHA256SUMS"
    if not path.exists():
        raise RuntimeError("SHA256SUMS is required")
    listed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        normalized = Path(relative.lstrip("* ")).as_posix()
        assert normalized not in listed, f"duplicate checksum entry: {normalized}"
        listed.add(normalized)
        target = HERE / normalized
        assert sha256(target) == expected, f"checksum mismatch: {target}"
    actual = {
        item.relative_to(HERE).as_posix()
        for item in released_files()
    }
    assert listed == actual, (
        f"checksum inventory mismatch: missing={sorted(actual - listed)}, "
        f"extra={sorted(listed - actual)}"
    )


def validate_replayed_screen(manifest: dict):
    """Check the retained full-size, NPZ-only numerical replay."""

    record = manifest["release_evidence"]
    path = MANIFEST_PATH.parent / record["paper_profile_numerical_screen"]
    assert sha256(path) == record["paper_profile_numerical_screen_sha256"]
    replay = json.loads(path.read_text(encoding="utf-8"))
    assert replay["schema"] == "frozen-candidate-numerical-screen-v1"
    assert replay["profile"] == "paper"
    for name, candidate in manifest["candidates"].items():
        result = replay["candidates"][name]
        assert result["points"] == 2**20
        assert result["margin_violations_inside"] == 0
        assert result["strict_violations_inside"] == 0
        assert result["nonfinite_evaluations"] == 0
        assert result["sampled_boundary"]["contained"]
        assert result["sampled_boundary"]["points_inside_or_on_level"] == 0
        if name == "grune10d":
            near_origin = result["near_origin_log_radial"]
            assert near_origin["points"] == 2**18
            assert near_origin["coverage"] == 1.0
            assert near_origin["minimum_radius"] == 1e-12
            assert near_origin["maximum_radius"] == 1e-3
            assert near_origin["margin_violations_inside"] == 0
            assert near_origin["strict_violations_inside"] == 0
            assert near_origin["nonfinite_evaluations"] == 0
            assert near_origin["maximum_scaled_log_derivative_inside"] < -0.01
        expected_coverage = (
            candidate["iid_radial_coverage"]
            if name == "grune10d"
            else candidate["independent_coverage"]
        )
        assert result["coverage"] == expected_coverage


def main():
    if not __debug__:
        raise RuntimeError(
            "validate_release.py must not be run with -O because its checks use assert"
        )
    manifest = load_manifest(MANIFEST_PATH)
    assert manifest["schema"] == "frozen-anchored-lyapunov-candidates-v1"
    assert candidate_names(MANIFEST_PATH) == ("cubic", "grune10d", "two_machine")
    for name, metadata in manifest["candidates"].items():
        validate_candidate(name, metadata)
    validate_replayed_screen(manifest)
    validate_sha256sums()
    print("PASS: three frozen candidates, portable weights, evidence, and checksums")


if __name__ == "__main__":
    main()
