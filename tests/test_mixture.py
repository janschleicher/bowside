import numpy as np
import pytest

from bowside.mixture import GaussianMixture, GaussianMixtureTimeseries


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


def test_gaussian_mixture_timeseries():
    sequences = np.array(
        [
            [[0.0, 0.0], [0.2, 0.1], [0.1, 0.3], [2.0, 2.0], [2.2, 2.1], [1.9, 2.3]],
            [[1.0, 1.0], [1.2, 1.1], [1.1, 1.3], [3.0, 3.0], [3.2, 3.1], [2.9, 3.3]],
        ],
        dtype=float,
    )
    time_points = np.array([0.0, 1.0], dtype=float)
    cluster_labels = np.array([0, 0, 0, 1, 1, 1], dtype=int)

    model = GaussianMixtureTimeseries(
        sequences=sequences,
        time_points=time_points,
        cluster_labels=cluster_labels,
    )

    # Check model attributes
    assert model.n_components == 2, "The model should discover two components."
    assert model.time_points.tolist() == [0.0, 1.0], "Time points should be sorted."
    assert model.weights.shape == (2,), "Weights should have one value per component."
    assert np.isclose(model.weights.sum(), 1.0), "Weights should sum to 1.0."
    assert model.means.shape == (2, 2, 2), (
        "Means should track components and time points."
    )
    assert model.covariances.shape == (2, 2, 2, 2), (
        "Covariances should track components and time points."
    )
    assert len(model.gaussian_mixtures) == 2, (
        "There should be one mixture per time point."
    )

    # Check PDFs
    samples = [[0.1, 0.05], [2.1, 2.05], [1.1, 1.2]]
    sample_time_points = [0.0, 0.0, 1.0]
    component_pdfs = model.component_pdfs(samples, sample_time_points)
    assert component_pdfs.shape == (3, 2), (
        "Component PDFs should match samples and components."
    )
    component_pdfs_single = model.component_pdfs(samples[0], sample_time_points[0])
    assert component_pdfs_single.shape == (1, 2), (
        "Single-sample PDFs should still be 2D."
    )

    # Check component probabilities
    probabilities = model.predict_proba(samples, sample_time_points)
    assert probabilities.shape == (3, 2), (
        "Probabilities should match samples and components."
    )
    assert np.allclose(probabilities.sum(axis=1), 1.0), (
        "Probabilities should sum to 1.0 per sample."
    )
    assert np.all(probabilities.argmax(axis=1) == np.asarray([0, 1, 0])), (
        "Probabilities should be higher for the component closer to the sample."
    )

    # Check scores
    scores = model.score_samples(samples, sample_time_points)
    assert scores.shape == (3,), "Scores should be returned per sample."
    assert scores[0] > scores[1], (
        "The first sample should score higher near its cluster."
    )

    # Check exceptions
    with pytest.raises(
        ValueError,
        match="Parameter `sequences` must be 3-dimensional with shape "
        "\\(n_steps, n_sequences, n_var\\).",
    ):
        GaussianMixtureTimeseries(
            sequences=sequences[0, :, :],
            time_points=time_points,
            cluster_labels=cluster_labels,
        )
    with pytest.raises(
        ValueError, match="Parameter `time_points` must be 1-dimensional"
    ):
        GaussianMixtureTimeseries(
            sequences=sequences,
            time_points=np.array([[0.0, 1.0]], dtype=float),
            cluster_labels=cluster_labels,
        )
    with pytest.raises(
        ValueError, match="Parameter `cluster_labels` must be 1-dimensional."
    ):
        GaussianMixtureTimeseries(
            sequences=sequences,
            time_points=time_points,
            cluster_labels=cluster_labels[None, :],
        )
    with pytest.raises(
        ValueError, match="Shape mismatch between `time_points` and `sequences`."
    ):
        GaussianMixtureTimeseries(
            sequences=sequences,
            time_points=[0.0, 0.5, 1.0],
            cluster_labels=cluster_labels,
        )
    with pytest.raises(
        ValueError, match="Shape mismatch between `sequences` and `cluster_labels`."
    ):
        GaussianMixtureTimeseries(
            sequences=sequences,
            time_points=time_points,
            cluster_labels=cluster_labels[:-1],
        )
    with pytest.raises(
        ValueError, match="Mismatching number of samples and time points."
    ):
        model.component_pdfs(samples, [0.0, 1.0])
