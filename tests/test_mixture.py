import numpy as np
import pytest

from bowside.mixture import GaussianMixture


def test_gaussian_mixture():
    weights = np.array([0.25, 0.75], dtype=float)
    means = np.array([[-2.0], [2.0]], dtype=float)
    covariances = np.array([[[0.5]], [[0.5]]], dtype=float)

    model = GaussianMixture(weights=weights, means=means, covariances=covariances)

    assert model is not None, "Model should be instantiated successfully."
    assert model.weights.ndim == 1, "Weights should be 1-dimensional."
    assert np.all(model.weights >= 0.0), "All weights should be non-negative."
    assert np.isclose(model.weights.sum(), 1.0), "Weights should sum to 1.0."

    # Test that score_samples returns higher values near the means
    near = np.array([[-2.0]], dtype=float)
    far = np.array([[8.0]], dtype=float)

    near_score = model.score_samples(near)
    far_score = model.score_samples(far)

    assert near_score.item() > far_score.item(), (
        "Score should be higher near component mean."
    )

    # Test that score is higher at the first component mean
    point_mean = model.means[0].reshape(1, -1)
    point_away = point_mean + 10.0

    assert (
        model.score_samples(point_mean).item() > model.score_samples(point_away).item()
    ), "Score should be higher at component mean than far away."

    # Test component probabilities
    prob = model.predict_proba([-1.5])
    assert np.isclose(prob.sum(), 1.0), "Component probabilities should sum to 1."
    assert prob[:, 0] > prob[:, 1], (
        "Probability should be higher for the first component."
    )

    probs = model.predict_proba([[0.0], [2.0], [2.2], [-2.5], [-1.8]])
    assert np.all(probs.argmax(axis=1) == np.asarray([1, 1, 1, 0, 0])), (
        "Incorrect component assignments."
    )


def test_gaussian_mixture_exceptions():
    weights = np.array([0.25, 0.75], dtype=float)
    means = np.array([[-2.0], [2.0]], dtype=float)
    covariances = np.array([[[0.5]], [[0.5]]], dtype=float)

    # Test weights must be 1-dimensional
    with pytest.raises(ValueError, match="Weights must be 1-dimensional"):
        GaussianMixture(
            weights=np.array([[0.25, 0.75]], dtype=float),
            means=means,
            covariances=covariances,
        )

    # Test means must be 2-dimensional
    with pytest.raises(ValueError, match="Means must be 2-dimensional"):
        GaussianMixture(
            weights=weights,
            means=np.array([-2.0, 2.0], dtype=float),
            covariances=covariances,
        )

    # Test covariances must be 3-dimensional
    with pytest.raises(ValueError, match="Covariances must be 3-dimensional"):
        GaussianMixture(
            weights=weights,
            means=means,
            covariances=np.array([[0.5, 0.5]], dtype=float),
        )

    # Test all weights must be positive
    with pytest.raises(ValueError, match="All weights must be positive"):
        GaussianMixture(
            weights=np.array([-0.25, 1.25], dtype=float),
            means=means,
            covariances=covariances,
        )

    # Test weights must sum to 1
    with pytest.raises(ValueError, match="Weights must sum to 1"):
        GaussianMixture(
            weights=np.array([0.25, 0.5], dtype=float),
            means=means,
            covariances=covariances,
        )

    # Test weights and means have mismatching shapes
    with pytest.raises(ValueError, match="Weights and means have mismatching shapes"):
        GaussianMixture(
            weights=np.array([0.5, 0.5], dtype=float),
            means=np.array([[-2.0], [2.0], [0.0]], dtype=float),
            covariances=covariances,
        )

    # Test weights and covariances have mismatching shapes
    with pytest.raises(
        ValueError, match="Weights and covariances have mismatching shapes"
    ):
        GaussianMixture(
            weights=np.array([0.5, 0.5], dtype=float),
            means=means,
            covariances=np.array([[[0.5]]], dtype=float),
        )

    # Test means and covariances have mismatching feature dimensions
    with pytest.raises(
        ValueError, match="Means and covariances have mismatching shapes"
    ):
        GaussianMixture(
            weights=weights,
            means=means,
            covariances=np.array(
                [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]], dtype=float
            ),
        )

    # Test covariance matrices must be square
    with pytest.raises(ValueError, match="Covariance matrices are not square"):
        GaussianMixture(
            weights=weights,
            means=np.array([[-2.0, -2.0], [2.0, 2.0]], dtype=float),
            covariances=np.array(
                [
                    [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                    [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                ],
                dtype=float,
            ),
        )
