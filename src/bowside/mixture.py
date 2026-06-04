"""Module for mixture models used in trajectory inference."""

from collections.abc import Collection
from typing import Union

import numpy as np
import pandas as pd
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
        Array-like of shape (n_components, n_var) containing the
        mean vectors of the mixture components.
    covariances : np.typing.ArrayLike
        Array-like of shape (n_components, n_var, n_var)
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
        self.n_var = self.means.shape[1]
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
            Array of shape (n_samples, n_var). A 1-dimensional array of
            shape (n_var,) can be passed as well for a single sample.

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
            Array of shape (n_samples, n_var). A 1-dimensional array of
            shape (n_var,) can be passed as well for a single sample.

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
            Array of shape (n_samples, n_var). A 1-dimensional array of
            shape (n_var,) can be passed as well for a single sample.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_components) containing the
            log-likelihood for each sample given the mixture model.
        """
        component_density = self.component_pdfs(X)
        return np.log((component_density * self.weights).sum(axis=1))


class GaussianMixtureTimeseries:
    """
    A (pseudo)time series of Gaussian mixtures describing expression trajectories.

    Parameters
    ----------
    sequences : np.typing.ArrayLike
        Array of shape (n_sequences, n_steps, n_var) containing gene
        expression sequences.
    time_points : np.typing.ArrayLike
        Array of shape (n_steps,) containing the time points.
    cluster_labels : np.typing.ArrayLike
        An array of shape (n_sequences). The value at index i is the
        cluster number to which sequence i belongs.
    allow_singular : bool, optional
        Whether to allow a singular covariance matrix. Default: False.

    Raises
    ------
    ValueError
        An error is raised if any of the input arrays is of the wrong shape.
    """

    def __init__(
        self,
        sequences: np.typing.ArrayLike,
        time_points: np.typing.ArrayLike,
        cluster_labels: np.typing.ArrayLike,
        allow_singular: bool = False,
    ) -> None:
        sequences = np.asarray(sequences)
        time_points = np.asarray(time_points)
        cluster_labels = np.asarray(cluster_labels)

        # Input checks
        if sequences.ndim != 3:
            raise ValueError(
                "Parameter `sequences` must be 3-dimensional with shape "
                "(n_steps, n_sequences, n_var)."
            )
        if time_points.ndim != 1:
            raise ValueError("Parameter `time_points` must be 1-dimensional.")
        if cluster_labels.ndim != 1:
            raise ValueError("Parameter `cluster_labels` must be 1-dimensional.")
        if time_points.shape[0] != sequences.shape[1]:
            raise ValueError("Shape mismatch between `time_points` and `sequences`.")
        if sequences.shape[0] != cluster_labels.shape[0]:
            raise ValueError("Shape mismatch between `sequences` and `cluster_labels`.")

        time_points.sort()
        self.time_points = pd.Index(time_points)

        # Compute component means
        clusters, weights = np.unique(
            cluster_labels,
            return_counts=True,
        )
        self.weights = weights / weights.sum(keepdims=True)

        # Compute cluster means and covariances for each time point
        self.n_components = clusters.shape[0]
        self.means = np.zeros(
            (self.n_components, time_points.shape[0], sequences.shape[2]), dtype=float
        )
        self.covariances = np.zeros(
            (
                self.n_components,
                time_points.shape[0],
                sequences.shape[2],
                sequences.shape[2],
            )
        )
        for i, cluster in enumerate(clusters):
            cluster_sequences = sequences[cluster_labels == cluster, :, :]
            self.means[i] = cluster_sequences.mean(axis=0)
            for t in range(time_points.shape[0]):
                self.covariances[i, t, :, :] = np.cov(
                    cluster_sequences[:, t, :], bias=False, rowvar=False
                )

        # Generate Gaussian mixtures
        self.gaussian_mixtures = [
            GaussianMixture(
                self.weights,
                self.means[:, t, :],
                self.covariances[:, t, :, :],
                allow_singular=allow_singular,
            )
            for t in range(time_points.shape[0])
        ]

    def component_pdfs(
        self,
        X: np.typing.ArrayLike,
        time_points: Union[int, float, Collection[Union[int, float]]],
    ) -> np.ndarray:
        """
        Evaluate the components' PDF for each sample, using the Gaussian
        mixture of the respective time point.

        Parameters
        ----------
        X : np.typing.ArrayLike
            Array of shape (n_samples, n_var). A 1-dimensional array of
            shape (n_var,) can be passed as well for a single sample.
        time_points : Union[int, float, Collection[Union[int, float]]]
            List containing the time points for each sample. If just one
            sample is given, a single numeric value can be given for the
            corresponding time point as well.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_components) containing the PDF
            value for each sample and component, given the time points.

        Raises
        ------
        ValueError
            An error is raised if the number of time points does not
            match the number of samples.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if isinstance(time_points, (int, float)):
            time_points = [time_points]
        if len(time_points) != X.shape[0]:
            raise ValueError("Mismatching number of samples and time points.")
        time_point_idx = np.asarray(
            [self.time_points.get_loc(tp) for tp in time_points]
        )
        pdfs = np.zeros((X.shape[0], self.n_components))
        for tp_index in np.unique(time_point_idx):
            pdfs[time_point_idx == tp_index, :] = self.gaussian_mixtures[
                tp_index
            ].component_pdfs(X[time_point_idx == tp_index, :])
        return pdfs

    def predict_proba(
        self,
        X: np.typing.ArrayLike,
        time_points: Union[int, float, Collection[Union[int, float]]],
    ) -> np.ndarray:
        """
        Predict the components' probabilities for each sample, using the
        Gaussian mixture of the respective time point.

        Parameters
        ----------
        X : np.typing.ArrayLike
            Array of shape (n_samples, n_var). A 1-dimensional array of
            shape (n_var,) can be passed as well for a single sample.
        time_points : Union[int, float, Collection[Union[int, float]]]
            List containing the time points for each sample. If just one
            sample is given, a single numeric value can be given for the
            corresponding time point as well.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_components) containing the
            probabilities for each sample and component, given the time
            points.
        """
        component_density = self.component_pdfs(X, time_points)
        weighted_density = component_density * self.weights
        return weighted_density / weighted_density.sum(axis=1, keepdims=True)

    def score_samples(
        self,
        X: np.typing.ArrayLike,
        time_points: Union[int, float, Collection[Union[int, float]]],
    ) -> np.ndarray:
        """
        Compute the log-likelihood of each sample, using the Gaussian
        mixture of the respective time point.

        Parameters
        ----------
        X : np.typing.ArrayLike
            Array of shape (n_samples, n_var). A 1-dimensional array of
            shape (n_var,) can be passed as well for a single sample.
        time_points : Union[int, float, Collection[Union[int, float]]]
            List containing the time points for each sample. If just one
            sample is given, a single numeric value can be given for the
            corresponding time point as well.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_components) containing the
            log-likelihood for each sample given the mixture model of
            the respective time points.
        """
        component_density = self.component_pdfs(X, time_points)
        return np.log((component_density * self.weights).sum(axis=1))
