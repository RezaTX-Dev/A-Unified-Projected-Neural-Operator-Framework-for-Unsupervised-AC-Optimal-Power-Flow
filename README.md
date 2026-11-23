# Randomized AC-OPF Neural Operator Framework

## Overview

This repository contains the implementation of a unified projected neural operator framework for AC Optimal Power Flow (AC-OPF).

The dataset is generated through fresh AC-OPF solutions under randomized load perturbations rather than relying on pre-solved datasets.

## Features

- Randomized active and reactive load perturbations
- Fresh AC-OPF solutions for dataset generation
- Multiple IEEE benchmark systems
- Neural operator architectures
- Feasibility-oriented projection framework
- Comparative evaluation of learning approaches

## Supported Test Systems

- IEEE 14-bus
- IEEE 30-bus
- IEEE 57-bus
- IEEE 118-bus
- IEEE 300-bus

## Requirements

```bash
pip install torch numpy scipy pandas pypower
```

## Running the Code

Example:

```bash
python main.py --cases case14 --n-samples 20 --seeds 1 --epochs 5
```

For a larger dataset:

```bash
python main.py --cases case14 --n-samples 3000
```

## Dataset Generation

The program generates randomized load scenarios and solves each scenario using AC-OPF. Successful solutions are used to construct the training and evaluation datasets.

## Reproducibility

The experiments use configurable random seeds, dataset sizes, training parameters, and benchmark systems.

## Citation

If you use this code in your research, please cite the associated paper.

## License

MIT License.
