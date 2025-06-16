# GEMLSS Econometrics Package

This repository contains a minimal implementation of a GEMLSS (Generalized
Econometric Models for Location and Scale) framework in pure Python. The goal is
to provide a lightweight, dependency-free example of fitting a basic location
and scale model using gradient descent.

## Features
- Normal distribution log-likelihood
- Identity and log link functions
- Simple gradient descent optimizer
- `GEMLSSModel` class with `fit`, `predict`, and `loglikelihood` methods

## Usage
```python
from gemlss import GEMLSSModel

# example data
y = [1.0, 2.0, 1.5, 2.3]
X = [[1, 0], [1, 1], [1, 2], [1, 3]]  # intercept + feature

model = GEMLSSModel(iterations=5000, learning_rate=0.01)
model.fit(X, y)
print("Parameters", model.beta_mu, model.beta_sigma)
print("Log-likelihood", model.loglikelihood(X, y))
```

This example fits a normal GEMLSS model where both the mean and standard
deviation depend on a linear combination of the predictor.
