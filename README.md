# Convex Optimisation

Convex Optimisation coursework for Advanced Topics in Machine Learning (COMP0083) at UCL 2022

- Proximal Stochastic Gradient Algorithm (PSGA)
- Randomised Coordinate Proximal Gradient Algorithm (RCPGA)
- Fast Iterative Shrinkage Threshold Algorithm (FISTA)

To set up your python environment:

1. Install `poetry`

```shell
pip install poetry
```

2. Install dependencies

```shell
poetry install
```

## Recovering Sparse Features with LASSO:
<p align="center">
  <img width="46%" src="outputs/part_3/psga-x.png" />
  <img width="48.5%" src="outputs/part_3/rcpga-x.png" />
</p>
<p align="center">
  <em>Sparse Features from LASSO vs Actual Sparse Feature Vector</em>
</p>

## Support Vector Machine Decision Boundary with Dual Formulation:
<p align="center">
  <img width="48%" src="outputs/part_4/rcpga-contour.png" />
  <img width="48%" src="outputs/part_4/fista-contour.png" />
</p>
<p align="center">
  <em>Contour Plots for Half Moon's Problem</em>
</p>

<p align="center">
  <img width="48%" src="outputs/part_4/rcpga-loss.png" />
  <img width="46.5%" src="outputs/part_4/fista-loss.png" />
</p>
<p align="center">
  <em>Objective functions for RCPGA and FISTA</em>
</p>