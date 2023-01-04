import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from constants import DEFAULT_SEED, OUTPUTS_FOLDER
from src.gradient_algorithms import (
    ProximalStochasticGradientAlgorithm,
    RandomizedCoordinateProximalGradientAlgorithm,
)
from src.problems import SparseProblem


def plot_loss(
    losses: List[List[float]], labels: List[str], algorithm: str, save_path: str
):
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(5)
    for i, loss in enumerate(losses):
        plt.plot(loss, label=labels[i])
    if len(labels) > 1:
        plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Loss ({algorithm})")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_dimensions(
    x: np.ndarray, x_sparse: np.ndarray, algorithm: str, save_path: str
) -> None:
    """
    Plot dimensions of solution and actual sparse vector

    :param x: solution (number of dimension, 1)
    :param x_sparse: actual sparse vector (number of dimensions, 1)
    :param algorithm: algorithm generating the solution
    :param save_path: path to save figure
    """
    x_sparse = x_sparse.reshape(-1)
    x = x.reshape(-1)
    d = len(x)

    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(10)
    fig.set_figheight(5)
    # True variable x_sparse
    ax.stem(
        np.arange(d)[np.abs(x_sparse) > 0],
        x_sparse[np.abs(x_sparse) > 0],
        label="$x^*$",
    )
    ax.stem(
        np.arange(d)[np.abs(x) > 0],
        x[np.abs(x) > 0],
        label=r"$x_{\tau, \lambda}$",
        linefmt="k:",
        markerfmt="k^",
    )
    ax.axhline(0.0, color="red")
    ax.set_xlim([-10 + 0, d + 10])
    ax.set_ylim([-1.1, 1.1])
    ax.set_xlabel("index $i$")
    ax.set_ylabel("$x_i$")
    ax.legend()
    plt.title(f"Sparse Solution vs Actual ({algorithm})")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def part_3():
    np.random.seed(DEFAULT_SEED)

    # Part 3
    part_3_output_folder = os.path.join(OUTPUTS_FOLDER, "part_3")
    if not os.path.exists(part_3_output_folder):
        os.makedirs(part_3_output_folder)

    sparse_problem = SparseProblem.generate(
        number_of_samples=1000,
        number_of_dimensions=500,
        sparsity=50,
        std=0.06,
    )
    x0 = np.random.randn(sparse_problem.d).reshape(-1, 1)
    lambda_parameter = 10
    number_of_steps = 1000

    # Proximal Stochastic Gradient Algorithm
    proximal_stochastic_gradient_algorithm = ProximalStochasticGradientAlgorithm(
        sparse_problem,
        is_ergodic_mean=False,
    )
    x_psga, loss_psga = proximal_stochastic_gradient_algorithm.run(
        x=x0,
        lambda_parameter=lambda_parameter,
        number_of_steps=number_of_steps,
    )
    plot_dimensions(
        x=x_psga,
        x_sparse=sparse_problem.x_sparse,
        algorithm=f"Proximal Stochastic Gradient Algorithm (x), {lambda_parameter=}",
        save_path=os.path.join(part_3_output_folder, "psga-x"),
    )

    proximal_stochastic_gradient_algorithm.is_ergodic_mean = True
    x_psga_ergodic, loss_psga_ergodic = proximal_stochastic_gradient_algorithm.run(
        x=x0,
        lambda_parameter=lambda_parameter,
        number_of_steps=number_of_steps,
    )
    plot_dimensions(
        x=x_psga_ergodic,
        x_sparse=sparse_problem.x_sparse,
        algorithm=f"Proximal Stochastic Gradient Algorithm (x_bar), {lambda_parameter=}",
        save_path=os.path.join(part_3_output_folder, "psga-x-bar"),
    )

    plot_loss(
        losses=[loss_psga_ergodic, loss_psga],
        labels=["x_bar (ergodic mean)", "x"],
        algorithm=f"Proximal Stochastic Gradient Algorithm, {lambda_parameter=}",
        save_path=os.path.join(part_3_output_folder, "psga-loss"),
    )

    # Randomized Coordinate Proximal Gradient Algorithm
    randomized_coordinate_proximal_gradient_algorithm = (
        RandomizedCoordinateProximalGradientAlgorithm(sparse_problem)
    )
    x_rcpga, loss_rcpga = randomized_coordinate_proximal_gradient_algorithm.run(
        x=x0, lambda_parameter=lambda_parameter, number_of_steps=number_of_steps
    )
    plot_dimensions(
        x=x_rcpga,
        x_sparse=sparse_problem.x_sparse,
        algorithm=f"Randomized Coordinate Proximal Gradient Algorithm, {lambda_parameter=}",
        save_path=os.path.join(part_3_output_folder, "rcpga-x"),
    )
    plot_loss(
        losses=[loss_rcpga],
        labels=["x"],
        algorithm=f"Randomized Coordinate Proximal Gradient Algorithm, {lambda_parameter=}",
        save_path=os.path.join(part_3_output_folder, "rcpga-loss"),
    )


if __name__ == "__main__":
    part_3()
