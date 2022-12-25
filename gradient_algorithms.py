from problems import Problem
import numpy as np
from abc import ABC, abstractmethod


class GradientAlgorithm(ABC):
    def __init__(self, problem: Problem):
        self.problem = problem

    @abstractmethod
    def run(self, x, lambda_parameter, number_of_steps):
        pass


class ProximalStochasticGradientAlgorithm(GradientAlgorithm):
    def __init__(self, problem: Problem):
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

    def run(self, x, lambda_parameter, number_of_steps):
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
                x=x[j]
                - (gamma_j / self.problem.n)
                * self.problem.a_matrix_j(j)
                @ (
                    self.problem.a_matrix @ x - self.problem.y
                ),  # (number of dimensions, )
                gamma=gamma_j,
                lambda_parameter=lambda_parameter,
            )
        return x


class FastIterativeShrinkageThresholdAlgorithm(GradientAlgorithm):
    def __init__(self, problem: Problem):
        super().__init__(problem)

    @staticmethod
    def grad_f(a_matrix, x):
        return a_matrix @ x

    @staticmethod
    def calculate_new_t(t):
        return (1 + np.sqrt(1 + 4 * t**2)) / 2

    def calculate_gamma(self, lambda_parameter) -> float:
        pass

    def run(self, x, lambda_parameter, number_of_steps) -> np.ndarray:
        t = 1
        v = np.random.randn(self.problem.n)
        u = np.random.randn(self.problem.n)
        gamma = self.calculate_gamma(lambda_parameter)
        for _ in range(number_of_steps):
            u_new = self.problem.proximity_operator(
                x=v
                + gamma * self.problem.a_matrix @ self.grad_f(self.problem.a_matrix, v),
                gamma=gamma,
                lambda_parameter=lambda_parameter,
            )
            t_new = self.calculate_new_t(t)
            v_new = u + ((t - 1) / t_new) * (u_new - u)
            u = u_new
            v = v_new
            t = t_new
        return u
