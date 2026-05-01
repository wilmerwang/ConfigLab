# ConfigLab

![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

ConfigLab is a config-driven deep learning template for reproducible experiments.
It combines Hydra configuration composition with PyTorch Lightning training loops,
using an MNIST reference pipeline to make experimentation, debugging, and
hyperparameter search fast to iterate on.

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Why It Is Useful](#why-it-is-useful)
- [How To Get Started](#how-to-get-started)
- [Project Layout](#project-layout)
- [Where To Get Help](#where-to-get-help)
- [Who Maintains and Contributes](#who-maintains-and-contributes)
- [License](#license)

## What This Project Does

ConfigLab provides a modular training pipeline where components are assembled from
configurations instead of hardcoded wiring.

- Uses Hydra to compose configs for data, model parts, callbacks, loggers, and trainer settings.
- Uses Lightning to run train/test/predict stages with one shared pipeline runner.
- Ships with configurable encoder/head options for an MNIST classification example.
- Stores reproducibility artifacts including git snapshots and tracked diffs in run outputs.

In short: you can switch behavior by changing config groups and command-line
overrides, not by rewriting your training script.

## Why It Is Useful

- Reproducible experiments by design
	- Hydra creates structured run directories and ConfigLab snapshots git state per run.
- Fast experimentation
	- Swap encoder/logger/trainer choices via overrides like `encoder=mlp` or `logger=csv`.
- Clear separation of concerns
	- Data, model, training, and runtime settings live in dedicated config groups.
- Built-in workflow presets
	- `task/train`, `task/test`, `task/predict`, debug presets, and Optuna sweeps are already scaffolded.
- Contributor-friendly codebase
	- Tests cover config composition, data loaders, model components, and utility helpers.

## How To Get Started

### 1) Prerequisites

- Python 3.13+
- `uv` (recommended package manager/runtime)

### 2) Install

```bash
git clone <your-fork-or-repo-url>
cd ConfigLab
uv sync --dev
```

Optional environment variable (defaults to current directory if unset):

```bash
export PROJECT_ROOT=$(pwd)
```

### 3) Run the pipeline

All commands below are run from the repository root.

Train and test in one run (from `configs/task/train.yaml`):

```bash
uv run python scripts/train.py task=train
```

Run testing only from a checkpoint:

```bash
uv run python scripts/test.py task=test ckpt_path="logs/train/runs/<timestamp>/checkpoints/<ckpt>.ckpt"
```

Run prediction from a checkpoint:

```bash
uv run python scripts/test.py task=predict ckpt_path="logs/train/runs/<timestamp>/checkpoints/<ckpt>.ckpt"
```

Switch model component via Hydra override (example: MLP encoder):

```bash
uv run python scripts/train.py task=train encoder=mlp
```

Run a debug profile (single epoch, CPU-safe settings):

```bash
uv run python scripts/train.py task=train debug=default
```

Run hyperparameter search with Optuna sweeper:

```bash
uv run python scripts/train.py task=train -m hparams_search=default
```

### 4) Validate locally

Run tests:

```bash
uv run pytest
```

Run linting (if Ruff is available in your environment):

```bash
uv run ruff check .
```

### 5) Understand outputs

Hydra writes run artifacts under:

- `logs/<task_name>/runs/<YYYY-MM-DD_HH-MM-SS>/` for single runs
- `logs/<task_name>/multiruns/<YYYY-MM-DD_HH-MM-SS>/` for sweeps

Typical artifacts include:

- checkpoint files
- task logs
- git snapshot data (`git_snapshot/`)

## Project Layout

```text
configs/            # Hydra composition and runtime presets
scripts/            # Entry points (train/test)
src/configlab/      # Library code (configs, data, models, pipeline, utils)
tests/              # Unit tests for configs/data/models/utils
data/               # Local datasets (MNIST)
logs/               # Generated experiment outputs
```

## Where To Get Help

- Open an issue in this repository for bugs or feature requests.
- Check [README.md](README.md), [CHANGELOG.md](CHANGELOG.md), and the config presets under [configs](configs/) for expected usage.
- For framework references used by this project:
	- Hydra: <https://hydra.cc/docs/intro/>
	- PyTorch Lightning: <https://lightning.ai/docs/pytorch/stable/>

## Who Maintains and Contributes

The project is currently maintained by the author listed in [LICENSE](LICENSE): `ikun`.

Contributions are welcome.

Suggested contribution workflow:

1. Fork the repository and create a feature branch.
2. Implement your change with clear, minimal scope.
3. Add or update tests in `tests/` when behavior changes.
4. Run `uv run pytest` locally.
5. Open a pull request with a concise summary and rationale.

If you are proposing larger structural changes (new config groups, training
pipelines, or data modules), include a short design note in the pull request
description.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
