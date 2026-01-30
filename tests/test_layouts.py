"""Tests for comparing layout optimization backends."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

from umapjax.layouts import jax as jax_layout

try:
    from umapjax.layouts import mlx as mlx_layout

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mlx_layout = None

try:
    from umapjax.layouts import torch as torch_layout

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch_layout = None

# Typical UMAP parameters (a, b) computed from min_dist=0.1
UMAP_A = 1.576943
UMAP_B = 0.8950608


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


def _create_structured_test_data(n_samples: int = 200, k: int = 15, seed: int = 42):
    """Create structured test data with k-NN graph for layout optimization.

    Returns
    -------
    tuple
        (initial_embedding, head, tail, weight)
    """
    X, _ = make_blobs(n_samples=n_samples, n_features=10, centers=4, random_state=seed)
    initial_embedding = np.random.default_rng(seed).standard_normal((n_samples, 2)).astype(np.float32)

    # Create edges based on k-nearest neighbors
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nn.kneighbors(X)

    head = np.repeat(np.arange(n_samples), k).astype(np.int32)
    tail = indices.flatten().astype(np.int32)
    weight = (1.0 / (1.0 + distances.flatten())).astype(np.float32)

    return initial_embedding, head, tail, weight


def _run_backend(backend_module, head_embedding, head, tail, weight, n_epochs, seed):
    """Run a layout optimization backend."""
    rng_state = np.array([seed], dtype=np.int64)
    return backend_module.optimize_layout_euclidean(
        head_embedding=head_embedding.copy(),
        tail_embedding=head_embedding.copy(),
        head=head.copy(),
        tail=tail.copy(),
        weight=weight.copy(),
        n_epochs=n_epochs,
        a=UMAP_A,
        b=UMAP_B,
        rng_state=rng_state,
        gamma=1.0,
        initial_alpha=1.0,
        negative_sample_rate=5,
        batch_size=128,
    )


def test_jax_produces_valid_output():
    """Test that JAX backend produces valid output."""
    head_embedding, head, tail, weight = _create_structured_test_data()
    result = _run_backend(jax_layout, head_embedding, head, tail, weight, n_epochs=10, seed=42)

    assert result.shape == head_embedding.shape
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
def test_mlx_produces_valid_output():
    """Test that MLX backend produces valid output."""
    head_embedding, head, tail, weight = _create_structured_test_data()
    result = _run_backend(mlx_layout, head_embedding, head, tail, weight, n_epochs=10, seed=42)

    assert result.shape == head_embedding.shape
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_torch_produces_valid_output():
    """Test that PyTorch backend produces valid output."""
    head_embedding, head, tail, weight = _create_structured_test_data()
    result = _run_backend(torch_layout, head_embedding, head, tail, weight, n_epochs=10, seed=42)

    assert result.shape == head_embedding.shape
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
def test_jax_mlx_neighbor_overlap():
    """Test that JAX and MLX backends produce similar neighbor structure."""
    head_embedding, head, tail, weight = _create_structured_test_data()
    k = 15

    result_jax = _run_backend(jax_layout, head_embedding, head, tail, weight, n_epochs=100, seed=42)
    result_mlx = _run_backend(mlx_layout, head_embedding, head, tail, weight, n_epochs=100, seed=42)

    overlap = _compute_overlap(result_jax, result_mlx, k)
    assert overlap > 0.5, f"JAX-MLX neighbor overlap {overlap:.3f} is too low"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_jax_torch_neighbor_overlap():
    """Test that JAX and PyTorch backends produce similar neighbor structure."""
    head_embedding, head, tail, weight = _create_structured_test_data()
    k = 15

    result_jax = _run_backend(jax_layout, head_embedding, head, tail, weight, n_epochs=100, seed=42)
    result_torch = _run_backend(torch_layout, head_embedding, head, tail, weight, n_epochs=100, seed=42)

    overlap = _compute_overlap(result_jax, result_torch, k)
    assert overlap > 0.5, f"JAX-Torch neighbor overlap {overlap:.3f} is too low"


@pytest.mark.skipif(not HAS_MLX or not HAS_TORCH, reason="MLX or PyTorch not available")
def test_mlx_torch_neighbor_overlap():
    """Test that MLX and PyTorch backends produce similar neighbor structure."""
    head_embedding, head, tail, weight = _create_structured_test_data()
    k = 15

    result_mlx = _run_backend(mlx_layout, head_embedding, head, tail, weight, n_epochs=100, seed=42)
    result_torch = _run_backend(torch_layout, head_embedding, head, tail, weight, n_epochs=100, seed=42)

    overlap = _compute_overlap(result_mlx, result_torch, k)
    assert overlap > 0.5, f"MLX-Torch neighbor overlap {overlap:.3f} is too low"


@pytest.mark.skipif(not HAS_MLX or not HAS_TORCH, reason="MLX or PyTorch not available")
def test_all_backends_similar_to_jax():
    """Test that all backends produce embeddings with similar neighbor structure to JAX."""
    head_embedding, head, tail, weight = _create_structured_test_data()
    k = 15

    result_jax = _run_backend(jax_layout, head_embedding, head, tail, weight, n_epochs=100, seed=42)
    result_mlx = _run_backend(mlx_layout, head_embedding, head, tail, weight, n_epochs=100, seed=42)
    result_torch = _run_backend(torch_layout, head_embedding, head, tail, weight, n_epochs=100, seed=42)

    # All should produce finite results
    assert np.all(np.isfinite(result_jax))
    assert np.all(np.isfinite(result_mlx))
    assert np.all(np.isfinite(result_torch))

    # Compare neighbor overlaps
    jax_mlx_overlap = _compute_overlap(result_jax, result_mlx, k)
    jax_torch_overlap = _compute_overlap(result_jax, result_torch, k)

    assert jax_mlx_overlap > 0.5, f"JAX-MLX overlap {jax_mlx_overlap:.3f} is too low"
    assert jax_torch_overlap > 0.5, f"JAX-Torch overlap {jax_torch_overlap:.3f} is too low"
