import chex
import numpy as np
import pytest
import umap
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

import umapjax


def test_package_has_version():
    assert umapjax.__version__ is not None


def test_initialization():
    """Test flexible initialization."""
    # Test valid init
    model = umapjax.UmapJax(n_neighbors=10, min_dist=0.5, n_epochs=200)
    assert model.n_neighbors == 10
    assert model.min_dist == 0.5
    assert model.n_epochs == 200


def test_init_errors():
    """Test expected errors during initialization."""
    # Test non-euclidean output metric
    with pytest.raises(NotImplementedError, match="Only euclidean output metric is implemented"):
        umapjax.UmapJax(output_metric="cosine")

    # Test densmap not implemented
    with pytest.raises(NotImplementedError, match="Densmap is not implemented"):
        umapjax.UmapJax(densmap=True)


def test_method_errors():
    """Test expected errors for unimplemented methods."""
    model = umapjax.UmapJax()
    dummy_data = np.random.rand(10, 5)

    with pytest.raises(NotImplementedError):
        model.transform(dummy_data)

    with pytest.raises(NotImplementedError):
        model.inverse_transform(dummy_data)

    with pytest.raises(NotImplementedError):
        model.update(dummy_data)


def test_fit_transform_shape_and_type():
    """Test basic output properties."""
    X = np.random.rand(100, 10)
    model = umapjax.UmapJax(n_neighbors=5, n_epochs=50)
    embedding = model.fit_transform(X)

    chex.assert_type(embedding, np.float32)
    chex.assert_shape(embedding, (100, 2))


def _compute_overlap(emb1: np.ndarray, emb2: np.ndarray, k: int) -> float:
    """Compute the mean overlap of k-nearest neighbors between two embeddings."""
    nn1 = NearestNeighbors(n_neighbors=k).fit(emb1)
    _, indices1 = nn1.kneighbors(emb1)

    nn2 = NearestNeighbors(n_neighbors=k).fit(emb2)
    _, indices2 = nn2.kneighbors(emb2)

    overlaps = []
    for i in range(len(emb1)):
        set1 = set(indices1[i])
        set2 = set(indices2[i])
        intersection = len(set1.intersection(set2))
        overlaps.append(intersection / k)
    return float(np.mean(overlaps))


def test_batch_size_parameter():
    """Test that batch_size parameter is respected and runs without error."""
    X = np.random.rand(100, 10)
    model_small = umapjax.UmapJax(n_neighbors=5, batch_size=10)
    embedding_small = model_small.fit_transform(X)
    assert embedding_small.shape == (100, 2)

    model_large = umapjax.UmapJax(n_neighbors=5, batch_size=100)
    embedding_large = model_large.fit_transform(X)
    assert embedding_large.shape == (100, 2)

    # Check that the embeddings are somewhat similar regardless of batch size
    overlap = _compute_overlap(embedding_small, embedding_large, k=5)
    assert overlap > 0.65, f"Overlap {overlap} is too low"


@pytest.mark.parametrize("layout_backend", ["jax", "mx", "torch"])
@pytest.mark.parametrize("spectral_backend", ["jax", "torch", "scipy"])
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_comparison_with_umap_learn(metric: str, layout_backend: str, spectral_backend: str):
    """Compare UmapJax with umap-learn by checking nearest neighbor overlap."""
    # Generate synthetic data with structure
    X, _ = make_blobs(n_samples=500, n_features=20, centers=5, random_state=42)
    k = 15

    # Run Reference UMAP (Seed 42)
    ref_model_1 = umap.UMAP(n_neighbors=k, n_epochs=200, metric=metric, random_state=42)
    embedding_ref_1 = ref_model_1.fit_transform(X)

    # Run Reference UMAP (Seed 43) for Baseline
    ref_model_2 = umap.UMAP(n_neighbors=k, n_epochs=200, metric=metric, random_state=43)
    embedding_ref_2 = ref_model_2.fit_transform(X)

    # Run UmapJax (Seed 42)
    jax_model = umapjax.UmapJax(
        n_neighbors=k,
        n_epochs=200,
        metric=metric,
        random_state=42,
        layout_backend=layout_backend,
        spectral_backend=spectral_backend,
    )
    embedding_jax = jax_model.fit_transform(X)

    # Compute Baselines
    baseline_overlap = _compute_overlap(embedding_ref_1, embedding_ref_2, k)
    jax_overlap = _compute_overlap(embedding_ref_1, embedding_jax, k)

    chex.assert_trees_all_close(
        jax_overlap,
        baseline_overlap,
        atol=0.1,
        rtol=0.1,
    )
