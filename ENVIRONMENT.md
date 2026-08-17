# Tested environment

The frozen artifact was replayed on:

- Ubuntu 22.04.5 LTS, Linux 5.15, x86-64;
- two Intel Xeon Gold 6542Y sockets, 48 physical cores / 96 hardware threads;
- 1 TiB RAM;
- four NVIDIA H100 NVL GPUs;
- Python 3.10.12;
- NumPy 2.2.3, SciPy 1.15.2, PyTorch 2.6.0+cu124, Matplotlib 3.10.0,
  and FontTools 4.57.0;
- dReal 4.21.6.2 at precision `1e-4`, with polytope-for-forall and local
  optimization enabled.

The reported exact-system standalone formal replays used CPU execution, were
launched concurrently, and used 48 dReal jobs each. The two-machine and cubic
runs took 1044.80 and 346.19 seconds in total, respectively. The corresponding
outer-shell queries took 1035.00 and 333.36 seconds. These are hardware- and
load-dependent timing observations, not part of the certificate.

The bundled paper-profile numerical replay was run on a GPU; an independent
CPU replay exactly recovered its coverages and zero-violation counts. CUDA is
not required for integrity validation, plotting, screening, or dReal
verification. It is required only for `verify_planar_crown.py`.

dReal is a native dependency and is intentionally not listed as an ordinary
PyPI requirement.  Install version 4.21.6.2, including its Python bindings,
from the dReal project before invoking `verify_planar.py`.  All other release
commands, except the optional GPU verifier, use the versions pinned in
`requirements.txt`.

## GPU CROWN environment

The retained GPU verification runs used α,β-CROWN 0.7.0 at commit
`6b8bbcfac1c01da1cabd240a87e4dce1a65f5a2b`, PyTorch 2.6.0+cu124, and one
NVIDIA H100 NVL. To reproduce that software revision in an environment with a
CUDA-capable PyTorch installation:

```bash
git clone --recursive \
  https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN
git checkout 6b8bbcfac1c01da1cabd240a87e4dce1a65f5a2b
git submodule update --init --recursive
python -m pip install .
```

Run `verify_planar_crown.py` from this repository after activating the same
environment. These verification scripts use CROWN bound propagation followed
by exhaustive adaptive input subdivision; they do not use MIP, GCP-CROWN, or
β-CROWN activation branching, so no commercial solver license is needed.
The retained CROWN runs use float32 without a proved roundoff envelope and are
therefore corroborating evidence; the dReal `UNSAT` runs are the
real-arithmetic formal certificates.

The retained GPU CROWN verification portions took 20.32 seconds for the
two-machine candidate and 53.42 seconds for the cubic-core candidate. The
in-process elapsed times recorded in the evidence JSON were 25.06 and 57.91
seconds, respectively. These are hardware- and load-dependent observations,
not part of the certificates.
