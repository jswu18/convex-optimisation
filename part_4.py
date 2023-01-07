import os

import matplotlib.pyplot as plt
import numpy as np

from constants import DEFAULT_SEED, OUTPUTS_FOLDER
from helpers import plot_loss
from src.gradient_algorithms import (
    FastIterativeShrinkageThresholdAlgorithm,
    RandomizedCoordinateProximalGradientAlgorithm,
)
from src.problems import HalfMoonsProblem


def plot_contour(
    half_moons_problem: HalfMoonsProblem,
    x: np.ndarray,
    contour_resolution: int,
    algorithm: str,
    save_path: str,
) -> None:
    x_min = np.min(half_moons_problem.x[:, 0]) - 0.1
    x_max = np.max(half_moons_problem.x[:, 0]) + 0.1
    y_min = np.min(half_moons_problem.x[:, 1]) - 0.1
    y_max = np.max(half_moons_problem.x[:, 1]) + 0.1
    x_ticks = np.linspace(
        x_min,
        x_max,
        contour_resolution,
    )
    y_ticks = np.linspace(
        y_min,
        y_max,
        contour_resolution,
    )

    x1_grid, x2_grid = np.meshgrid(x_ticks, y_ticks)
    x_grid = np.stack((x1_grid.flatten(), x2_grid.flatten())).T

    y_grid = half_moons_problem.predict(x.T, x_grid).reshape(-1)
    y = half_moons_problem.y.reshape(-1)
    fig, ax = plt.subplots(constrained_layout=True)
    fig.set_figwidth(3.5 * (x_max - x_min))
    fig.set_figheight(3.5 * (y_max - y_min))
    cmap = plt.get_cmap("binary_r", 2)
    contour = ax.contourf(x1_grid, x2_grid, y_grid.reshape(x1_grid.shape), cmap=cmap)
    for label in set(y):
        edge = "gray"
        if label == -1:
            colour = "black"
        else:
            colour = "white"
        idx = y == label
        plt.scatter(
            half_moons_problem.x[idx, 0],
            half_moons_problem.x[idx, 1],
            label=label,
            c=colour,
            edgecolors=edge,
            s=50,
        )
    plt.legend()
    fig.colorbar(contour, ticks=[-1, 1])
    plt.title(f"Contour Plot ({algorithm})")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def part_4(
    contour_resolution: int,
):
    part_4_output_folder = os.path.join(OUTPUTS_FOLDER, "part_4")
    if not os.path.exists(part_4_output_folder):
        os.makedirs(part_4_output_folder)

    number_of_samples = 200
    sigma = 0.275
    lambda_parameter = 3e-3
    x0 = np.random.randn(number_of_samples).reshape(-1, 1)

    half_moons_problem = HalfMoonsProblem.generate(
        number_of_samples=number_of_samples,
        noise=0.05,
        sigma=sigma,
        random_state=0,
    )
    plot_contour(
        half_moons_problem=half_moons_problem,
        x=x0,
        contour_resolution=contour_resolution,
        algorithm=f"Initial Contour, {sigma=}",
        save_path=os.path.join(part_4_output_folder, "initial-contour"),
    )

    # Randomized Coordinate Projected Gradient Algorithm
    number_of_steps = int(3e5)
    rcpga = RandomizedCoordinateProximalGradientAlgorithm(half_moons_problem)
    x_rcpga, loss_rcpga = rcpga.run(x0.copy(), lambda_parameter, number_of_steps)
    plot_contour(
        half_moons_problem=half_moons_problem,
        x=x_rcpga,
        contour_resolution=contour_resolution,
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
    number_of_steps = 20
    fista = FastIterativeShrinkageThresholdAlgorithm(half_moons_problem)
    x_fista, loss_fista = fista.run(x0.copy(), lambda_parameter, number_of_steps)
    plot_contour(
        half_moons_problem=half_moons_problem,
        x=x_fista,
        contour_resolution=contour_resolution,
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
    np.random.seed(DEFAULT_SEED)
    CONTOUR_RESOLUTION = 250
    part_4(
        contour_resolution=CONTOUR_RESOLUTION,
    )
