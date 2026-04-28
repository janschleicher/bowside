"""Module for mixture models used in trajectory inference."""

import numpy as np
from scipy.stats import multivariate_normal


class GaussianMixture:
    """
    A multivariate Gaussian mixture model with frozen weights, means,
    and covariances.

    Parameters
    ----------
    weights : np.typing.ArrayLike
        Array-like of shape (n_components,) containing the weights of
        the mixture components.
    means : np.typing.ArrayLike
        Array-like of shape (n_components, n_features) containing the
        mean vectors of the mixture components.
    covariances : np.typing.ArrayLike
        Array-like of shape (n_components, n_features, n_features)
        containing the covariance matrices of the mixture components.
    allow_singular : bool, optional
        Whether to allow a singular covariance matrix. Default: False.

    Raises
    ------
    ValueError
        An error is raised if any of the input arrays is of the wrong shape.
    """

    def __init__(
        self,
        weights: np.typing.ArrayLike,
        means: np.typing.ArrayLike,
        covariances: np.typing.ArrayLike,
        allow_singular: bool = False,
    ) -> None:
        self.weights = np.asarray(weights, dtype=float)
        self.means = np.asarray(means, dtype=float)
        self.covariances = np.asarray(covariances, dtype=float)

        # Check input
        if self.weights.ndim != 1:
            raise ValueError("Weights must be 1-dimensional.")
        if self.means.ndim != 2:
            raise ValueError("Means must be 2-dimensional.")
        if self.covariances.ndim != 3:
            raise ValueError("Covariances must be 3-dimensional.")
        if np.any(self.weights < 0.0):
            raise ValueError("All weights must be positive.")
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("Weights must sum to 1.")
        if self.means.shape[0] != self.weights.shape[0]:
            raise ValueError("Weights and means have mismatching shapes.")
        if self.covariances.shape[0] != self.weights.shape[0]:
            raise ValueError("Weights and covariances have mismatching shapes.")
        if self.covariances.shape[1] != self.means.shape[1]:
            raise ValueError("Means and covariances have mismatching shapes.")
        if self.covariances.shape[1] != self.covariances.shape[2]:
            raise ValueError("Covariance matrices are not square.")

        # Initialize components
        self.n_components = self.means.shape[0]
        self.n_features = self.means.shape[1]
        self.components = [
            multivariate_normal(
                mean=self.means[i],
                cov=self.covariances[i],
                allow_singular=allow_singular,
            )
            for i in range(self.n_components)
        ]

    def component_pdfs(self, X: np.typing.ArrayLike) -> np.ndarray:
        """
        Evaluate the components' PDF for each sample.

        Parameters
        ----------
        X : np.typing.ArrayLike
            Array of shape (n_samples, n_features). A 1-dimensional array of
            shape (n_features,) can be passed as well for a single sample.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_components) containing the PDF
            value for each sample and component.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        return np.column_stack([component.pdf(X) for component in self.components])

    def predict_proba(self, X: np.typing.ArrayLike) -> np.ndarray:
        """
        Predict the components' probabilities for each sample.

        Parameters
        ----------
        X : np.typing.ArrayLike
            Array of shape (n_samples, n_features). A 1-dimensional array of
            shape (n_features,) can be passed as well for a single sample.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_components) containing the
            probabilities for each sample and component.
        """
        component_density = self.component_pdfs(X)
        weighted_density = component_density * self.weights
        return weighted_density / weighted_density.sum(axis=1, keepdims=True)

    def score_samples(self, X: np.typing.ArrayLike) -> np.ndarray:
        """
        Compute the log-likelihood of each sample.

        Parameters
        ----------
        X : np.typing.ArrayLike
            Array of shape (n_samples, n_features). A 1-dimensional array of
            shape (n_features,) can be passed as well for a single sample.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_components) containing the
            log-likelihood for each sample given the mixture model.
        """
        component_density = self.component_pdfs(X)
        return np.log((component_density * self.weights).sum(axis=1))
