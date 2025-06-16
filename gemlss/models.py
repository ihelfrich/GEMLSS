"""GEMLSS model implementation."""

import math
from typing import Callable, List, Sequence

from .links import IdentityLink, LogLink
from .distributions import Normal


class GEMLSSModel:
    """Simple GEMLSS model for normal data with location and scale.

    Parameters are modeled as linear functions of predictors transformed by link
    functions.
    """

    def __init__(
        self,
        location_link: Callable[[float], float] = IdentityLink(),
        scale_link: Callable[[float], float] = LogLink(),
        learning_rate: float = 0.001,
        iterations: int = 1000,
    ):
        self.location_link = location_link
        self.scale_link = scale_link
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.beta_mu: List[float] = []
        self.beta_sigma: List[float] = []

    @staticmethod
    def _dot(x: Sequence[float], beta: Sequence[float]) -> float:
        return sum(x_i * b_i for x_i, b_i in zip(x, beta))

    def _initialize(self, p: int):
        self.beta_mu = [0.0 for _ in range(p)]
        self.beta_sigma = [0.0 for _ in range(p)]

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]):
        """Fit the model to the data using gradient descent."""
        if not X:
            raise ValueError("Empty design matrix")
        p = len(X[0])
        self._initialize(p)
        n = len(y)

        for _ in range(self.iterations):
            grad_mu = [0.0] * p
            grad_sigma = [0.0] * p
            for xi, yi in zip(X, y):
                mu = self.location_link.inverse(self._dot(xi, self.beta_mu))
                sigma = self.scale_link.inverse(self._dot(xi, self.beta_sigma))
                if sigma <= 0:
                    sigma = 1e-6
                resid = yi - mu
                grad_common = resid / (sigma ** 2)
                for j in range(p):
                    grad_mu[j] += grad_common * xi[j]
                    grad_sigma[j] += ((resid ** 2 - sigma ** 2) / sigma ** 3) * xi[j]
            # update
            for j in range(p):
                self.beta_mu[j] += self.learning_rate * grad_mu[j] / n
                self.beta_sigma[j] += self.learning_rate * grad_sigma[j] / n

    def predict(self, X: Sequence[Sequence[float]]):
        """Predict mean and scale for given design matrix."""
        preds = []
        for xi in X:
            mu = self.location_link.inverse(self._dot(xi, self.beta_mu))
            sigma = self.scale_link.inverse(self._dot(xi, self.beta_sigma))
            preds.append((mu, max(sigma, 1e-6)))
        return preds

    def loglikelihood(self, X: Sequence[Sequence[float]], y: Sequence[float]):
        ll = 0.0
        for (mu, sigma), yi in zip(self.predict(X), y):
            dist = Normal(mu, sigma)
            ll += dist.logpdf(yi)
        return ll
