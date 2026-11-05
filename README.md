# deepRD

**Stochastic reaction-diffusion simulations meet deep learning.**

deepRD is a Python package for simulating molecular dynamics and stochastic reaction-diffusion processes, and for learning reduced models of them with PyTorch. Detailed molecular or particle simulations produce training data; deep generative models learn the unresolved noise and interaction terms; and reduced Langevin integrators sample those terms on the fly to reproduce the full dynamics at a fraction of the cost (stochastic closure).

## Deep learning samplers (PyTorch)

The core of deepRD's data-driven modeling lives in `deepRD.noiseSampler`:

- **`cvaeSampler`**: a conditional variational autoencoder that learns the distribution of the noise term conditioned on the particle state (e.g. velocity and previous noise values), for bistable and dimer systems.
- **`diffusionSampler`**: a conditional diffusion model with adaptive layer norm (AdaLN) blocks, supporting several conditioning schemes on velocities, relative positions and noise history.
- **`binning`**: a non-parametric baseline that samples from binned trajectory data.

Training code is in `deepRD/noiseSampler/training/`. Models run on GPU automatically when CUDA is available. Trained samplers plug directly into the data-driven integrators `langevinNoiseSampler` and `langevinInteractionSampler`.

## Simulation tools

- **Diffusion integrators:** Langevin (BAOAB, ABOBA, semi-implicit Euler), overdamped Langevin, and Smoluchowski dynamics with a reactive boundary and particle reservoir
- **Reaction integrators:** Gillespie SSA and tau-leaping
- **Reaction models:** genetic feedback, Schlögl, reservoir
- **Potentials:** harmonic, bistable, Gaussian, and pair bistable (dimer) potentials
- **Tools:** trajectory, particle and analysis utilities, plus plotting helpers

## Installation

```bash
git clone https://github.com/maojrs/deepRD.git
cd deepRD
pip install -e .
pip install numpy scipy matplotlib joblib torch jupyter
```

## Quick start

```python
import deepRD
from deepRD.diffusionIntegrators import langevin

particle = deepRD.particle([0, 0, 0], velocity=[0, 0, 0], mass=1)
particleList = deepRD.particleList([particle])

# dt, stride, tfinal, Gamma (friction), kBT, boxsize
integrator = langevin(0.01, 1, 10, 10.0, 1, 1)
t, X, V = integrator.propagate(particleList)
```

```python
from deepRD.reactionModels import geneticFeedback
from deepRD.reactionIntegrators import gillespie

model = geneticFeedback(G=1, Gstar=0, M=50, P=300)
t, X = gillespie(stride=1, tfinal=10).propagate(model)
```

## Repository structure

```
deepRD/
├── deepRD/
│   ├── noiseSampler/          # PyTorch CVAE and diffusion samplers, binning, training code
│   ├── diffusionIntegrators/  # Langevin, Smoluchowski and data-driven integrators
│   ├── reactionIntegrators/   # Gillespie and tau-leap
│   ├── reactionModels/        # Genetic feedback, Schlögl, reservoir
│   ├── potentials/            # External and pair potentials
│   └── tools/                 # Trajectory, particle and analysis utilities
├── notebooks/                 # Examples and analyses
└── scripts/                   # Data generation and benchmarks
```

## Examples

The `notebooks/` folder contains worked examples:

- `stochasticClosureCVAE/`: benchmarks of reduced models driven by the deep generative samplers
- `stochasticClosure/`: the binning-based closure pipeline, from building the sampling model to comparing reduced and reference dynamics
- `openSystems/`: Smoluchowski dynamics coupled to a particle reservoir
- `langevin.ipynb`, `geneticFeedbackModel.ipynb` and others for the basic integrators

Matching batch scripts for data generation and benchmarking, including first-passage-time analyses, are in `scripts/`.
