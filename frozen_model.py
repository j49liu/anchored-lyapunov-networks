"""Portable evaluation of the three frozen Lyapunov candidates.

Only NumPy arrays and a small JSON manifest are loaded.  The public evaluator
does not unpickle a training checkpoint and does not import training code.

Every candidate has the same anchored form

    H(x) = V_p(x) exp(gain * (N(x / scale) - N(0))),
    W(x) = tanh(H(x)).

All three cores are positive degree-two homogeneous tanh models. The
correction ``N`` is a standard tanh network with a scalar linear output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "artifacts" / "candidates.json"


class FrozenLyapunov(nn.Module):
    """An anchored Lyapunov candidate reconstructed from portable arrays."""

    def __init__(self, metadata: dict, arrays: dict[str, np.ndarray]):
        super().__init__()
        self.metadata = metadata
        self.name = metadata["name"]
        self.dimension = int(metadata["dimension"])
        self.degree = int(metadata["homogeneous_degree"])
        self.core_kind = metadata["core_kind"]
        self.correction_gain = float(metadata["correction_gain"])
        self.latent_level = float(metadata["latent_level"])
        self.bounded_level = float(metadata["bounded_level"])

        if self.core_kind == "homogeneous_tanh":
            self.register_buffer("core_feature_weight", _tensor(arrays["core_feature_weight"]))
            self.register_buffer("core_feature_bias", _tensor(arrays["core_feature_bias"]))
            self.register_buffer("core_output_weight", _tensor(arrays["core_output_weight"]))
            self.register_buffer("core_output_bias", _tensor(arrays["core_output_bias"]))
        elif self.core_kind == "quadratic":
            self.register_buffer("core_matrix", _tensor(arrays["core_matrix"]))
        else:
            raise ValueError(f"unsupported core kind: {self.core_kind}")

        self.register_buffer("input_scale", _tensor(arrays["input_scale"]).reshape(1, -1))
        depth = int(metadata["correction_depth"])
        self.correction_weights = nn.ParameterList()
        self.correction_biases = nn.ParameterList()
        for index in range(depth):
            self.correction_weights.append(
                nn.Parameter(_tensor(arrays[f"correction_weight_{index}"]), requires_grad=False)
            )
            self.correction_biases.append(
                nn.Parameter(_tensor(arrays[f"correction_bias_{index}"]), requires_grad=False)
            )
        self.register_buffer("correction_output_weight", _tensor(arrays["correction_output_weight"]))

    def correction_network(self, states: torch.Tensor) -> torch.Tensor:
        hidden = states
        for weight, bias in zip(self.correction_weights, self.correction_biases):
            hidden = torch.tanh(torch.nn.functional.linear(hidden, weight, bias))
        return torch.nn.functional.linear(hidden, self.correction_output_weight)

    def correction(self, states: torch.Tensor) -> torch.Tensor:
        normalized = states / self.input_scale
        origin = torch.zeros((1, self.dimension), dtype=states.dtype, device=states.device)
        return self.correction_gain * (
            self.correction_network(normalized) - self.correction_network(origin)
        )

    def core_value(self, states: torch.Tensor) -> torch.Tensor:
        if self.core_kind == "quadratic":
            return torch.einsum("bi,ij,bj->b", states, self.core_matrix, states).unsqueeze(1)
        radius = torch.linalg.vector_norm(states, dim=1, keepdim=True)
        safe_radius = radius.clamp_min(torch.finfo(states.dtype).tiny)
        unit = states / safe_radius
        hidden = torch.tanh(
            torch.nn.functional.linear(
                unit, self.core_feature_weight, self.core_feature_bias
            )
        )
        log_shape = torch.nn.functional.linear(
            hidden,
            self.core_output_weight.reshape(1, -1),
            self.core_output_bias.reshape(1),
        )
        value = radius**self.degree * torch.exp(log_shape)
        return torch.where(radius > 0, value, torch.zeros_like(value))

    def latent(self, states: torch.Tensor) -> torch.Tensor:
        return self.core_value(states) * torch.exp(self.correction(states))

    def bounded(self, states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.latent(states))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.bounded(states)

    def log_gradient(self, states: torch.Tensor) -> torch.Tensor:
        """Evaluate the exact analytic gradient of ``log H`` off the origin."""

        if states.ndim != 2 or states.shape[1] != self.dimension:
            raise ValueError("states must have shape (batch, dimension)")
        if torch.any(torch.linalg.vector_norm(states, dim=1) == 0):
            raise ValueError("log H and its gradient are undefined at the origin")

        if self.core_kind == "quadratic":
            value = self.core_value(states)
            core_gradient = torch.nn.functional.linear(
                states, self.core_matrix + self.core_matrix.T
            ) / value
        else:
            radius = torch.linalg.vector_norm(states, dim=1, keepdim=True)
            unit = states / radius
            hidden = torch.tanh(
                torch.nn.functional.linear(
                    unit, self.core_feature_weight, self.core_feature_bias
                )
            )
            angular = (
                self.core_output_weight * (1.0 - hidden.square())
            ) @ self.core_feature_weight
            tangent = angular - (angular * unit).sum(dim=1, keepdim=True) * unit
            core_gradient = (self.degree * unit + tangent) / radius

        normalized = states / self.input_scale
        hidden = normalized
        jacobian = torch.diag_embed(1.0 / self.input_scale.expand(len(states), -1))
        for weight, bias in zip(self.correction_weights, self.correction_biases):
            hidden = torch.tanh(torch.nn.functional.linear(hidden, weight, bias))
            jacobian = torch.einsum("oi,bij->boj", weight, jacobian)
            jacobian = (1.0 - hidden.square()).unsqueeze(-1) * jacobian
        correction_gradient = torch.einsum(
            "oi,bij->boj", self.correction_output_weight, jacobian
        ).squeeze(1)
        return core_gradient + self.correction_gain * correction_gradient


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array)).detach().clone()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_candidate(name: str, manifest_path: Path = DEFAULT_MANIFEST) -> FrozenLyapunov:
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    try:
        metadata = manifest["candidates"][name]
    except KeyError as error:
        choices = ", ".join(sorted(manifest["candidates"]))
        raise KeyError(f"unknown candidate {name!r}; choose from {choices}") from error
    weight_path = (manifest_path.parent / metadata["portable_weights"]).resolve()
    with np.load(weight_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    return FrozenLyapunov(metadata, arrays).eval()


def candidate_names(manifest_path: Path = DEFAULT_MANIFEST) -> tuple[str, ...]:
    return tuple(sorted(load_manifest(manifest_path)["candidates"]))
