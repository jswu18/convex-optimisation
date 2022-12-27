from abc import ABC, abstractmethod

import numpy as np

from problems import Problem


class GradientAlgorithm(ABC):
    def __init__(self, problem: Problem):
        self.problem = problem

    @abstractmethod
    def run(self, x, lambda_parameter, number_of_steps):
        pass


class ProximalStochasticGradientAlgorithm(GradientAlgorithm):
    def __init__(self, problem: Problem, is_ergodic_mean: bool):
        self.is_ergodic_mean = is_ergodic_mean
        super().__init__(problem)

    def generate_random_data_index(self):
        return np.random.choice(self.problem.n)

    def calculate_gamma(self, k):
        """
        A matrix (number of points, number of dimensions)
        k: step number
        """
        return self.problem.n / (
            np.linalg.norm(self.problem.a_matrix) ** 2 * np.sqrt(k + 1)
        )

    def _run_ergodic_mean(self, x, lambda_parameter, number_of_steps):
        gamma_normaliser = 0
        x_bar_unnormalised = np.zeros(x.shape)
        for k in range(number_of_steps):
            i = self.generate_random_data_index()
            gamma_k = self.calculate_gamma(k)
            x = self.problem.proximity_operator(
                x=x
                - gamma_k
                * (self.problem.a_matrix_i(i) @ x - self.problem.y[i])
                * self.problem.a_matrix_i(i),  # (number of dimensions, )
                gamma=gamma_k,
                lambda_parameter=lambda_parameter,
            )
            gamma_normaliser += gamma_k
            x_bar_unnormalised += gamma_k * x
        return x_bar_unnormalised / gamma_normaliser

    def _run_not_ergodic_mean(self, x, lambda_parameter, number_of_steps):
        for k in range(number_of_steps):
            i = self.generate_random_data_index()
            gamma_k = self.calculate_gamma(k)
            x = self.problem.proximity_operator(
                x=x
                - gamma_k
                * (self.problem.a_matrix_i(i) @ x - self.problem.y[i])
                * self.problem.a_matrix_i(i),  # (number of dimensions, )
                gamma=gamma_k,
                lambda_parameter=lambda_parameter,
            )
        return x

    def run(self, x, lambda_parameter, number_of_steps):
        if self.is_ergodic_mean:
            return self._run_ergodic_mean(x, lambda_parameter, number_of_steps)
        else:
            return self._run_not_ergodic_mean(x, lambda_parameter, number_of_steps)


class RandomizedCoordinateProximalGradientAlgorithm(GradientAlgorithm):
    def __init__(self, problem: Problem):
        super().__init__(problem)

    def generate_random_dimension_index(self):
        return np.random.choice(self.problem.d)

    def calculate_gamma(self, j):
        """
        A matrix (number of points, number of dimensions)
        j: dimension
        """
        return self.problem.n / (np.linalg.norm(self.problem.a_matrix_j(j)) ** 2)

    def run(self, x, lambda_parameter, number_of_steps):
        for k in range(number_of_steps):
            j = self.generate_random_dimension_index()
            gamma_j = self.calculate_gamma(j)
            x[j] = self.problem.proximity_operator(
                x=x[j] - (gamma_j / self.problem.n) * self.problem.grad_f_j(x, j),
                gamma=gamma_j,
                lambda_parameter=lambda_parameter,
            )
        return x


class FastIterativeShrinkageThresholdAlgorithm(GradientAlgorithm):
    def __init__(self, problem: Problem):
        super().__init__(problem)

    @staticmethod
    def calculate_new_t(t):
        return (1 + np.sqrt(1 + 4 * t**2)) / 2

    def calculate_gamma(self, lambda_parameter) -> float:
        return 2 / (np.linalg.norm(self.problem.a_matrix) ** 2)

    def run(self, x, lambda_parameter, number_of_steps) -> np.ndarray:
        t = 1
        v = np.random.randn(self.problem.n)
        gamma = self.calculate_gamma(lambda_parameter)
        for _ in range(number_of_steps):
            x_new = self.problem.proximity_operator(
                x=v + gamma * self.problem.a_matrix @ self.problem.grad_f(v),
                gamma=gamma,
                lambda_parameter=lambda_parameter,
            )
            t_new = self.calculate_new_t(t)
            v_new = x + ((t - 1) / t_new) * (x_new - x)
            x = x_new
            v = v_new
            t = t_new
        return x
