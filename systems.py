"""Vector fields used by the frozen-certificate artifact.

The functions in this file are deliberately independent of LYZNet.  Their
PyTorch implementations are used for numerical screening and plotting; the
matching dReal expressions are written explicitly in ``verify_planar.py``.
"""

from __future__ import annotations

import math

import torch


PLANAR_DOMAINS = {
    "two_machine": ((-2.0, 3.0), (-3.0, 1.5)),
    "cubic": ((-1.5, 1.5), (-1.5, 1.5)),
}

DOMINANT_DEGREES = {"two_machine": 1, "cubic": 3, "grune10d": 1}


def two_machine(states: torch.Tensor) -> torch.Tensor:
    """Two-machine swing dynamics in equilibrium-centered coordinates."""

    x1, x2 = states.unbind(dim=-1)
    return torch.stack(
        (
            x2,
            -0.5 * x2
            - torch.sin(x1 + math.pi / 3.0)
            + math.sin(math.pi / 3.0),
        ),
        dim=-1,
    )


def two_machine_dominant(states: torch.Tensor) -> torch.Tensor:
    """Degree-one homogeneous part of :func:`two_machine`."""

    x1, x2 = states.unbind(dim=-1)
    return torch.stack((x2, -0.5 * x1 - 0.5 * x2), dim=-1)


def cubic(states: torch.Tensor) -> torch.Tensor:
    """Homogeneously dominated trigonometric cubic benchmark."""

    x1, x2 = states.unbind(dim=-1)
    radius_squared = x1.square() + x2.square()
    f1 = 4.0 * x1**3 - x1.square() * x2 - 6.0 * x1 * x2.square() - x2**3
    f2 = x1**3 + 4.0 * x1.square() * x2 + x1 * x2.square() - 6.0 * x2**3
    degree_five = 0.2 * radius_squared.square() * (
        3.0 + torch.sin(x1) + torch.cos(x2)
    )
    degree_seven = 0.05 * radius_squared**3 * (
        3.0 + torch.cos(x1 + x2) + torch.sin(x1 - x2)
    )
    modulation = degree_five + degree_seven
    return torch.stack((f1 + modulation * x1, f2 + modulation * x2), dim=-1)


def cubic_dominant(states: torch.Tensor) -> torch.Tensor:
    """Degree-three homogeneous part of :func:`cubic`."""

    x1, x2 = states.unbind(dim=-1)
    return torch.stack(
        (
            4.0 * x1**3 - x1.square() * x2 - 6.0 * x1 * x2.square() - x2**3,
            x1**3 + 4.0 * x1.square() * x2 + x1 * x2.square() - 6.0 * x2**3,
        ),
        dim=-1,
    )


def grune10d(states: torch.Tensor) -> torch.Tensor:
    """Untransformed ten-dimensional Gr\u00fcne/Liu polynomial benchmark."""

    x = states.unbind(dim=-1)
    return torch.stack(
        (
            -x[0] + 0.5 * x[1] - 0.1 * x[8].square(),
            -0.5 * x[0] - x[1],
            -x[2] + 0.5 * x[3] - 0.1 * x[0].square(),
            -0.5 * x[2] - x[3],
            -x[4] + 0.5 * x[5] + 0.1 * x[6].square(),
            -0.5 * x[4] - x[5],
            -x[6] + 0.5 * x[7],
            -0.5 * x[6] - x[7],
            -x[8] + 0.5 * x[9],
            -0.5 * x[8] - x[9] + 0.1 * x[1].square(),
        ),
        dim=-1,
    )


SYSTEMS = {
    "two_machine": two_machine,
    "cubic": cubic,
    "grune10d": grune10d,
}

DOMINANT_SYSTEMS = {
    "two_machine": two_machine_dominant,
    "cubic": cubic_dominant,
}

