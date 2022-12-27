from __future__ import annotations

import random
from abc import ABC, abstractmethod

import numpy as np
from sklearn.datasets import make_moons


class Problem(ABC):
    def __init__(self, a_matrix: np.ndarray, y: np.ndarray):
        """

        :param a_matrix: design matrix (number of points, number of dimensions)
        :param y: response vector (number of points, 1)
        """
        self.a_matrix = a_matrix
        self.y = y

    @property
    def n(self):
        return self.a_matrix.shape[0]

    @property
    def d(self):
        return self.a_matrix.shape[1]

    def a_matrix_i(self, i):
        return self.a_matrix[i, :]

    def a_matrix_j(self, j):
        return self.a_matrix[:, j]

    @staticmethod
    @abstractmethod
    def proximity_operator(x, gamma, lambda_parameter):
        pass

    @abstractmethod
    def loss(self, x, lambda_parameter: float):
        pass

    @staticmethod
    @abstractmethod
    def generate(*args, **kwargs) -> Problem:
        pass

    @abstractmethod
    def grad_f_j(self, x: np.ndarray, j: int) -> float:
        pass

    @abstractmethod
    def grad_f(self, x: np.ndarray) -> float:
        return self.a_matrix @ x


class SparseProblem(Problem):
    def __init__(
        self,
        a_matrix: np.ndarray,
        y: np.ndarray,
        x_sparse: np.ndarray,
        sparsity: int,
        std: float,
    ):
        self.x_sparse = x_sparse
        self.sparsity = sparsity
        self.std = std
        super().__init__(a_matrix, y)

    @staticmethod
    def proximity_operator(x, gamma, lambda_parameter):
        rho = gamma * lambda_parameter
        return np.sign(x - rho) * np.maximum(0, np.abs(x) - rho)

    def loss(self, x, lambda_parameter: float):
        return (1 / (2 * self.n)) * np.linalg.norm(
            self.a_matrix @ x - self.y
        ) ** 2 + lambda_parameter * np.linalg.norm(x, ord=1)

    def grad_f_j(self, x: np.ndarray, j) -> float:
        return self.a_matrix_j(j) @ (self.a_matrix @ x - self.y)

    def grad_f(self, x: np.ndarray) -> float:
        pass

    @staticmethod
    def generate(
        number_of_samples, number_of_dimensions, sparsity: int, std=0.06
    ) -> SparseProblem:
        """
        Generate xs vectors with entries in [0.5, 1] and [-1, -0.5] respectively.

        :param number_of_samples:
        :param number_of_dimensions:
        :param sparsity:
        :param std:
        :return:
        """

        assert sparsity % 2 == 0, f"{sparsity=} needs to be divisible by 2"
        xsp = 0.5 * (np.random.rand(sparsity // 2) + 1)
        xsn = -0.5 * (np.random.rand(sparsity // 2) + 1)
        x_sparse = np.hstack([xsp, xsn, np.zeros(number_of_dimensions - sparsity)])
        random.shuffle(x_sparse)

        # Generate A
        a_matrix = np.random.randn(number_of_samples, number_of_dimensions)

        # Generate eps
        y = a_matrix @ x_sparse + std * np.random.randn(number_of_samples)
        return SparseProblem(
            a_matrix=a_matrix,
            y=y,
            x_sparse=x_sparse,
            sparsity=sparsity,
            std=std,
        )


class HalfMoonsProblem(Problem):
    def __init__(self, x: np.ndarray, y: np.ndarray, sigma: float):
        self.x = x
        self.sigma = sigma
        self.gram_matrix = self.calculate_gram(x)
        super().__init__(
            a_matrix=np.multiply(y @ y.T, self.gram_matrix),
            y=y,
        )

    @staticmethod
    def kernel(x, y, sigma):
        diff = x - y
        return np.exp(-diff.T @ diff / (2 * sigma**2))

    def calculate_gram(self, x):
        n = self.x.shape[0]
        m = x.shape[0]
        gram_matrix = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                gram_matrix[i, j] = self.kernel(self.x[i, :], x[j, :], self.sigma)
        return gram_matrix

    @staticmethod
    def proximity_operator(x, gamma, lambda_parameter):
        rho = gamma / lambda_parameter
        out = np.copy(x)
        mask_1 = x > 1
        out[mask_1] = 0

        mask_2 = x < (1 - rho)
        out[mask_2] = rho

        out[~(mask_1 | mask_2)] = 1 - x[~(mask_1 | mask_2)]
        return out

    def loss(self, x, lambda_parameter: float):
        indicator = (
            np.float("inf")
            if np.sum(np.logical_or((x < 0), (x > (1 / (lambda_parameter * self.n)))))
            else 0
        )
        return 0.5 * x.T @ self.a_matrix @ x - np.sum(x) + indicator

    def grad_f_j(self, x: np.ndarray, j) -> float:
        return self.a_matrix_j(j) @ x

    def grad_f(self, x: np.ndarray) -> float:
        return self.a_matrix @ x

    @staticmethod
    def generate(
        number_of_samples: int, noise: float, sigma: float, random_state: int
    ) -> HalfMoonsProblem:
        x, y = make_moons(
            n_samples=number_of_samples, noise=noise, random_state=random_state
        )
        y = 2 * y - 1
        return HalfMoonsProblem(
            x=x,
            y=y.reshape(-1, 1),
            sigma=sigma,
        )

    def predict(self, alpha, x):
        gram_matrix = self.calculate_gram(x)
        return np.sign(
            gram_matrix.T @ np.multiply(self.y.reshape(-1, 1), alpha.reshape(-1, 1))
        ).reshape(-1)
