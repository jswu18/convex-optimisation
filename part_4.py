import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from constants import DEFAULT_SEED, OUTPUTS_FOLDER
from helpers import plot_loss
from src.gradient_algorithms import (
    FastIterativeShrinkageThresholdAlgorithm,
    RandomizedCoordinateProjectedGradientAlgorithm)
from src.problems import HalfMoonsProblem


def plot_contour(half_moons_problem, x, algorithm, save_path):
    x_min = np.min(half_moons_problem.x[:, 0]) - 1
    x_max = np.max(half_moons_problem.x[:, 0]) + 1
    y_min = np.min(half_moons_problem.x[:, 1]) - 1
    y_max = np.max(half_moons_problem.x[:, 1]) + 1
    x_ticks = np.linspace(
        x_min,
        x_max,
        100,
    )
    y_ticks = np.linspace(
        y_min,
        y_max,
        100,
    )

    x1_grid, x2_grid = np.meshgrid(x_ticks, y_ticks)
    x_grid = np.stack((x1_grid.flatten(), x2_grid.flatten())).T

    y_grid = half_moons_problem.predict(x.T, x_grid).reshape(-1)
    y = half_moons_problem.y.reshape(-1)
    fig = plt.figure()
    fig.set_figwidth(2 * (x_max - x_min))
    fig.set_figwidth(2 * (y_max - y_min))
    plt.contourf(x1_grid, x2_grid, y_grid.reshape(x1_grid.shape))
    for label in set(y):
        idx = y == label
        plt.scatter(
            half_moons_problem.x[idx, 0],
            half_moons_problem.x[idx, 1],
            label=label,
        )
    plt.legend()
    plt.title(f"Contour Plot ({algorithm})")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def part_4():
    np.random.seed(DEFAULT_SEED)

    # Part 4
    part_4_output_folder = os.path.join(OUTPUTS_FOLDER, "part_4")
    if not os.path.exists(part_4_output_folder):
        os.makedirs(part_4_output_folder)

    number_of_samples = 200
    x0 = np.random.randn(number_of_samples).reshape(-1, 1)
    sigma = 0.4
    lambda_parameter = 10
    number_of_steps = int(2e5)
    half_moons_problem = HalfMoonsProblem.generate(
        number_of_samples=number_of_samples,
        noise=0.05,
        sigma=sigma,
        random_state=0,
    )
    plot_contour(
        half_moons_problem=half_moons_problem,
        x=x0,
        algorithm=f"Initial Contour, {lambda_parameter=}, {sigma=}",
        save_path=os.path.join(part_4_output_folder, "initial-contour"),
    )

    # Randomized Coordinate Projected Gradient Algorithm
    rcpga = RandomizedCoordinateProjectedGradientAlgorithm(half_moons_problem)
    x_rcpga, loss_rcpga = rcpga.run(x0.copy(), lambda_parameter, number_of_steps)
    plot_contour(
        half_moons_problem=half_moons_problem,
        x=x_rcpga,
        algorithm=f"Randomized Coordinate Projected Gradient Algorithm, {lambda_parameter=}, {sigma=}",
        save_path=os.path.join(part_4_output_folder, "rcpga-contour"),
    )
    plot_loss(
        losses=[loss_rcpga],
        labels=["rcpga"],
        algorithm=f"Randomized Coordinate Projected Gradient Algorithm, {lambda_parameter=}, {sigma=}",
        save_path=os.path.join(part_4_output_folder, "rcpga-loss"),
    )

    # Fast Iterative Shrinkage Threshold Algorithm
    number_of_steps = 50
    fista = FastIterativeShrinkageThresholdAlgorithm(half_moons_problem)
    x_fista, loss_fista = fista.run(x0.copy(), lambda_parameter, number_of_steps)
    plot_contour(
        half_moons_problem=half_moons_problem,
        x=x_fista,
        algorithm=f"Fast Iterative Shrinkage Threshold Algorithm, {lambda_parameter=}, {sigma=}",
        save_path=os.path.join(part_4_output_folder, "fista-contour"),
    )
    plot_loss(
        losses=[loss_fista],
        labels=["fista"],
        algorithm=f"Fast Iterative Shrinkage Threshold Algorithm, {lambda_parameter=}, {sigma=}",
        save_path=os.path.join(part_4_output_folder, "fista-loss"),
    )


if __name__ == "__main__":
    part_4()
