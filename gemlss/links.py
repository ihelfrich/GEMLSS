"""Link functions for GEMLSS models."""

import math


class IdentityLink:
    """Identity link function."""

    def __call__(self, x: float) -> float:
        return x

    def inverse(self, y: float) -> float:
        return y


def _safe_exp(x: float, lim: float = 50.0) -> float:
    """Exponential with bounds to avoid overflow."""
    if x > lim:
        x = lim
    elif x < -lim:
        x = -lim
    return math.exp(x)


class LogLink:
    """Log link ensures positivity via exp."""

    def __call__(self, x: float) -> float:
        return math.log(x)

    def inverse(self, y: float) -> float:
        return _safe_exp(y)
