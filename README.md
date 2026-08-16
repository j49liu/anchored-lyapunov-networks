# Anchored Lyapunov network certificates

This lightweight artifact accompanies *Universal Approximation of Maximal
Lyapunov Functions with Anchored Neural Networks*. It contains three frozen
candidates and no training code:

- two planar candidates with independent dReal and GPU CROWN verification
  scripts; and
- one 10D candidate with a homogeneous neural anchor and retained empirical
  stress tests.

The artifact reproduces candidate evaluation, numerical screens, plots, and
formal verification from frozen weights.  It does **not** claim to reproduce
checkpoint training.  Candidate generation is not a contribution of the
paper.

The frozen candidates were trained using an experimental version of
[LyZNet](https://doi.org/10.1145/3641513.3650134).

## Installation

Use Python 3.10 on Linux and install the lightweight evaluation dependencies:

```bash
python -m pip install -r requirements.txt
```

dReal replay additionally requires dReal 4.21.6.2 with its Python bindings.
GPU replay requires a CUDA-capable PyTorch installation and α,β-CROWN 0.7.0;
α,β-CROWN is installed separately rather than hidden in the lightweight
requirements. See [`ENVIRONMENT.md`](ENVIRONMENT.md) for the exact tested
revision, installation commands, hardware, and replay times.

## Quick start

```bash
python validate_release.py
python screen_candidates.py --profile smoke
python plot_candidates.py
```

## Formal verification

Replay the six dReal queries for either planar candidate with:

```bash
timeout 3600s python verify_planar.py two_machine --jobs 48 \
  --output runs/two_machine.json
timeout 3600s python verify_planar.py cubic --jobs 48 \
  --output runs/cubic.json
```

Each formal run checks one complete unit-sphere obligation, one outer-shell
obligation, and the four faces of the verification box.  A candidate passes
only when every query is `UNSAT` and the analytic local bound closes.
The bundled standalone dReal records satisfy all six queries for both planar
candidates. dReal has no native per-query watchdog in this interface;
the `timeout` wrapper above fails externally if a candidate exceeds one hour.

The same two frozen candidates can be replayed on a CUDA GPU with:

```bash
python verify_planar_crown.py two_machine --device cuda \
  --output runs/two_machine_gpu_crown.json
python verify_planar_crown.py cubic --device cuda \
  --output runs/cubic_gpu_crown.json
```

This second verifier uses certified CROWN bounds from the α,β-CROWN software
and exhaustive adaptive subdivision of the input domain. It does not invoke
β-CROWN activation branching. A run succeeds only when every terminal tile is
certified; an unresolved tile, resource limit, or backend error fails closed.
The analytic local bound handles a ball around the origin, where the scaled
polar expression has a removable singularity.

On one NVIDIA H100 NVL, the retained CROWN verification portions took 20.32
seconds for the two-machine system and 53.42 seconds for the cubic-core
system. Timings are hardware dependent and are not part of the certificates.

## Numerical replay

The full numerical replay uses the retained population sizes:

```bash
python screen_candidates.py --profile paper --device cpu
```

The 10D result is an empirical stress test, not a formal certificate.  Its
reported `50.0161%` coverage is for the fixed IID radial proposal in the
radius-40 box. Its independent uniform-box coverage is `0.0394%`, illustrating
why the sampling measure must be stated in ten dimensions.

## Frozen candidates

| Candidate | Method | Fixed bounded level | Independent evidence |
|---|---|---:|---|
| Example VI.1 (two-machine system) | homogeneous neural anchor | `0.705906975` | 32.8486%, formally verified |
| Example VI.2 (cubic-core system) | homogeneous neural anchor | `0.041549706` | 19.9742%, formally verified |
| Example VI.3 (10D system) | homogeneous neural anchor | `0.395577875` | 50.0161% IID-radial, empirical |

The `.npz` files are the portable public representation used by evaluation,
plotting, and formal verification.  The original `weights_only=True` PyTorch
state dictionaries are included for provenance; only the integrity validator
loads them, to compare every source tensor with its NPZ copy.  `SHA256SUMS`
binds every released file.

## Scope

The paper-profile command replays the reported $2^{20}$-point IID-radial 10D
screen, a $2^{18}$-point near-origin test with logarithmically sampled radii
from $10^{-12}$ to $10^{-3}$, and a $2^{18}$-point sampled-boundary check.  The
larger archived 10D stress-test ledger contains additional populations,
attacks, and rollouts; it is retained as evidence but is not wholly regenerated
by this lightweight release.

The planar proofs establish strict decrease on the complete nonzero frozen
sublevel and separation from every face of the stated box.  The 10D plots are
coordinate sections of a ten-dimensional set; these sections are not
invariant subsystems.  The radius-40 box is a testing domain, not an asserted
basin enclosure.

The repository is distributed under the Apache-2.0 license in the project
root.
