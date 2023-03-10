from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from src.problems import HalfMoonsProblem, Problem, SparseProblem


class GradientAlgorithm(ABC):
    def __init__(self, problem: Problem):
        self.problem = problem

    @abstractmethod
    def run(
        self, x: np.ndarray, lambda_parameter: float, number_of_steps: int
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Run the gradient based algorithm.

        :param x: initial solution (number of dimension, 1)
        :param lambda_parameter: scalar on g(x) of the minimization problem
        :param number_of_steps: number of steps to run the algorithm
        :return: optimised solution from algorithm and list of loss values at each iteration
        """
        pass


class ProximalStochasticGradientAlgorithm(GradientAlgorithm):
    def __init__(self, problem: SparseProblem, is_ergodic_mean: bool):
        """
        The Proximal Stochastic Gradient Algorithm.

        :param problem: sparse problem
        :param is_ergodic_mean: whether to use the ergodic mean for loss calculation
        """
        self.is_ergodic_mean = is_ergodic_mean
        super().__init__(problem)

    def generate_random_data_index(self) -> int:
        """
        Random choice of a datapoint index from (1, ..., n)
        :return: randomly chosen index
        """
        return np.random.choice(self.problem.n)

    def calculate_gamma_parameter(self, k: int):
        """
        gamma_k = mu/(||A||^2 * sqrt (k+1))

        :param k: step number
        :return: gamma_k for the psga algorithm
        """

        return self.problem.mu / (
            np.linalg.norm(self.problem.a_matrix, ord=np.inf) ** 2 * np.sqrt(k + 1)
        )

    def run(
        self, x: np.ndarray, lambda_parameter: float, number_of_steps: int
    ) -> Tuple[np.ndarray, List[float]]:
        """
        The PSGA algorithm.

        :param x: initial solution (number of dimension, 1)
        :param lambda_parameter: scalar on g(x) of the minimization problem
        :param number_of_steps: number of steps to run the algorithm
        :return: optimised solution from PSGA and list of loss values at each iteration
        """
        gamma_normaliser = 0
        x_bar_unnormalised = np.zeros(x.shape)
        loss = [self.problem.loss(x, lambda_parameter)]
        for k in range(number_of_steps):
            i = self.generate_random_data_index()
            gamma_parameter_k = self.calculate_gamma_parameter(k)
            x = self.problem.proximity_operator(
                x=(
                    x  # (1, number of dimension)
                    - gamma_parameter_k
                    * (self.problem.a_matrix_i(i) @ x - self.problem.y[i])  # (1, 1)
                    * self.problem.a_matrix_i(i).T  # (number of dimensions, 1)
                ),
                gamma_parameter=gamma_parameter_k,
                lambda_parameter=lambda_parameter,
            )
            gamma_normaliser += gamma_parameter_k
            x_bar_unnormalised += gamma_parameter_k * x

            if self.is_ergodic_mean:
                loss_value = self.problem.loss(
                    x_bar_unnormalised / gamma_normaliser, lambda_parameter
                )
            else:
                loss_value = self.problem.loss(x, lambda_parameter)
            loss.append(loss_value)
        if self.is_ergodic_mean:
            x = x_bar_unnormalised / gamma_normaliser
        return x, loss


class RandomizedCoordinateProximalGradientAlgorithm(GradientAlgorithm):
    def __init__(self, problem: Problem):
        """
        The Randomized Coordinate Proximal Gradient Algorithm.

        :param problem: the problem
        """
        super().__init__(problem)

    def generate_random_dimension_index(self) -> int:
        """
        Random choice of a dimension index from (1, ..., d)
        :return: randomly chosen index
        """
        return np.random.choice(self.problem.d)

    def calculate_gamma_parameter(self, j: int) -> float:
        """
        gamma = mu/(||a_j||^2)
        where mu is the coefficient of strong convexity

        :param j: dimension index
        :return: gamma_j for the rcpga algorithm
        """
        return self.problem.mu / (np.linalg.norm(self.problem.a_matrix_j(j)) ** 2)

    def run(
        self, x, lambda_parameter, number_of_steps
    ) -> Tuple[np.ndarray, List[float]]:
        """
        The RCPGA algorithm.

        :param x: initial solution (number of dimension, 1)
        :param lambda_parameter: scalar on g(x) of the minimization problem
        :param number_of_steps: number of steps to run the algorithm
        :return: optimised solution from RCPGA and list of loss values at each iteration
        """
        loss = [self.problem.loss(x, lambda_parameter)]
        for k in range(number_of_steps):
            j = self.generate_random_dimension_index()
            gamma_parameter_j = self.calculate_gamma_parameter(j)
            x[j] = self.problem.proximity_operator(
                x=(
                    x[j]
                    - (gamma_parameter_j / self.problem.n) * self.problem.grad_f_j(x, j)
                ),
                gamma_parameter=gamma_parameter_j,
                lambda_parameter=lambda_parameter,
            )
            loss.append(self.problem.loss(x, lambda_parameter))
        return x, loss


class FastIterativeShrinkageThresholdAlgorithm(GradientAlgorithm):
    def __init__(self, problem: HalfMoonsProblem):
        """
        The Fast Iterative Shrinkage Threshold Algorithm.

        :param problem: half moons problem
        """
        super().__init__(problem)
        self.problem = problem

    @staticmethod
    def calculate_new_t(t: float) -> float:
        """
        t_new = (1+sqrt(1+4*t**2))/(2)
        :param t: current t value
        :return: new t value
        """
        return (1 + np.sqrt(1 + 4 * t**2)) / 2

    def calculate_gamma_parameter(self) -> float:
        """
        gamma = mu/L
        where L is the lipschitz constant of A^*A

        :return: gamma value
        """
        lipschitz_constant = np.linalg.eigvalsh(
            self.problem.a_matrix.T @ self.problem.a_matrix
        )[-1]
        return self.problem.mu / lipschitz_constant

    def run(
        self, x, lambda_parameter, number_of_steps
    ) -> Tuple[np.ndarray, List[float]]:
        """
        The FISTA Algorithm.

        :param x: initial solution (number of dimension, 1)
        :param lambda_parameter: scalar on g(x) of the minimization problem
        :param number_of_steps: number of steps to run the algorithm
        :return: optimised solution from FISTA and list of loss values at each iteration
        """
        t = 1
        v = np.random.uniform(
            low=0,
            high=1 / (lambda_parameter * self.problem.n),
            size=(self.problem.n,),
        ).reshape(-1, 1)
        gamma_parameter = self.calculate_gamma_parameter()
        loss = [self.problem.loss(x, lambda_parameter)]
        for _ in range(number_of_steps):
            x_new = self.problem.proximity_operator(
                x=v + gamma_parameter * self.problem.a_matrix @ self.problem.grad_f(x),
                gamma_parameter=gamma_parameter,
                lambda_parameter=lambda_parameter,
            )
            t_new = self.calculate_new_t(t)
            v_new = x + ((t - 1) / t_new) * (x_new - x)
            x = x_new
            v = v_new
            t = t_new
            loss.append(self.problem.loss(x, lambda_parameter))
        return x, loss
