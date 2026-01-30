import jax
import numpy as np
import pytest
import scipy.sparse
import umap.spectral

from umapjax.spectral.jax import spectral_layout as spectral_layout_jax
from umapjax.spectral.torch import spectral_layout as spectral_layout_torch

# Backend dispatch for parameterized tests
BACKENDS = {
    "jax": spectral_layout_jax,
    "torch": spectral_layout_torch,
}


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_spectral_layout_deterministic(backend):
    """Test that spectral layout is deterministic with same seed."""
    spectral_layout = BACKENDS[backend]

    n_samples = 50
    graph = scipy.sparse.random(n_samples, n_samples, density=0.1, random_state=42)
    graph = (graph + graph.T) / 2

    rng = np.random.default_rng(42)
    X = rng.random((n_samples, 10))
    dim = 2

    emb1 = spectral_layout(X, graph, dim, random_state=42)
    emb2 = spectral_layout(X, graph, dim, random_state=42)
    np.testing.assert_allclose(emb1, emb2)

    emb3 = spectral_layout(X, graph, dim, random_state=43)
    np.testing.assert_allclose(emb1, emb3, atol=1e-5)


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("graph_type", ["connected", "multi_component"])
def test_spectral_consistency(backend, graph_type):
    """Test consistency with umap-learn for connected and multi-component graphs."""
    spectral_layout = BACKENDS[backend]

    n_samples = 100
    dim = 2
    random_state = 42

    if graph_type == "connected":
        # Random connected graph
        graph = scipy.sparse.random(n_samples, n_samples, density=0.1, random_state=42)
        graph = (graph + graph.T) / 2
    else:
        # Multi-component graph
        row1 = np.arange(49)
        col1 = np.arange(1, 50)
        data1 = np.ones(49)
        row2 = np.arange(50, 99)
        col2 = np.arange(51, 100)
        data2 = np.ones(49)

        graph = scipy.sparse.coo_matrix(
            (
                np.hstack([data1, data1, data2, data2]),
                (np.hstack([row1, col1, row2, col2]), np.hstack([col1, row1, col2, row2])),
            ),
            shape=(n_samples, n_samples),
        )

    rng = np.random.default_rng(random_state)
    X = rng.random((n_samples, 10))

    emb_ref = umap.spectral.spectral_layout(
        data=X,
        graph=graph,
        dim=dim,
        random_state=random_state,
    )

    emb = spectral_layout(
        data=X,
        graph=graph,
        dim=dim,
        random_state=random_state,
    )

    # For multi-component, sign flips can happen per-component, so we must check per component.
    if graph_type == "connected":
        components = [np.arange(n_samples)]
    else:
        components = [np.arange(49), np.arange(50, 100)]

    for indices in components:
        comp_emb, comp_ref = emb[indices], emb_ref[indices]

        # Check centroids (meta-embedding consistency)
        np.testing.assert_allclose(comp_emb.mean(0), comp_ref.mean(0), atol=0.05)

        # Check centered component (spectral embedding consistency)
        c_emb, c_ref = comp_emb - comp_emb.mean(0), comp_ref - comp_ref.mean(0)
        assert c_emb.shape == c_ref.shape

        if graph_type == "connected":
            # Align signs by checking dot product of each dimension
            signs = np.sign(np.sum(c_emb * c_ref, axis=0))
            np.testing.assert_allclose(c_emb * signs, c_ref, atol=1e-1)
        else:
            # For multi-component, just ensure norms are somewhat clear (not zero)
            assert np.all(np.linalg.norm(c_emb, axis=0) > 1e-3)


@pytest.mark.parametrize("enable_x64", [True, False])
def test_spectral_layout_lobpcg_jax(monkeypatch, enable_x64):
    """Test JAX spectral layout using the LOBPCG path (forced by low threshold)."""
    jax.config.update("jax_enable_x64", enable_x64)

    rng = np.random.default_rng(42)
    n_samples = 100
    row = np.arange(n_samples)
    col = (np.arange(n_samples) + 1) % n_samples
    data = rng.uniform(size=(n_samples,))
    graph = scipy.sparse.coo_matrix((data, (row, col)), shape=(n_samples, n_samples))
    graph = graph + graph.T

    X = rng.random((n_samples, 10)).astype(np.float32)
    dim = 2

    # Patch the threshold to 0 to force LOBPCG path
    import umapjax.spectral.jax

    monkeypatch.setattr(umapjax.spectral.jax, "SPECTRAL_DENSE_THRESHOLD", 0)

    # Patch umap-learn's spectral_layout to support the 'method' argument
    monkeypatch.setattr(umap.spectral, "spectral_layout", umap.spectral._spectral_layout)

    emb = spectral_layout_jax(
        data=X,
        graph=graph,
        dim=dim,
        random_state=42,
    )

    # Output should be valid
    assert emb.shape == (n_samples, dim)
    assert not np.isnan(emb).any()

    emb_ref = umap.spectral.spectral_layout(data=X, graph=graph, dim=dim, random_state=42, method="lobpcg")
    # Align signs by checking dot product of each dimension
    signs = np.sign(np.sum(emb * emb_ref, axis=0))
    np.testing.assert_allclose(emb * signs, emb_ref, atol=4e-3)


def test_spectral_layout_lobpcg_torch(monkeypatch):
    """Test PyTorch spectral layout using the LOBPCG path (forced by low threshold)."""
    rng = np.random.default_rng(42)
    n_samples = 100
    row = np.arange(n_samples)
    col = (np.arange(n_samples) + 1) % n_samples
    data = rng.uniform(size=(n_samples,))
    graph = scipy.sparse.coo_matrix((data, (row, col)), shape=(n_samples, n_samples))
    graph = graph + graph.T

    X = rng.random((n_samples, 10)).astype(np.float32)
    dim = 2

    # Patch the threshold to 0 to force LOBPCG path
    import umapjax.spectral.torch

    monkeypatch.setattr(umapjax.spectral.torch, "SPECTRAL_DENSE_THRESHOLD", 0)

    # Patch umap-learn's spectral_layout to support the 'method' argument
    monkeypatch.setattr(umap.spectral, "spectral_layout", umap.spectral._spectral_layout)

    emb = spectral_layout_torch(
        data=X,
        graph=graph,
        dim=dim,
        random_state=42,
    )

    # Output should be valid
    assert emb.shape == (n_samples, dim)
    assert not np.isnan(emb).any()

    emb_ref = umap.spectral.spectral_layout(data=X, graph=graph, dim=dim, random_state=42, method="lobpcg")
    # Align signs by checking dot product of each dimension
    signs = np.sign(np.sum(emb * emb_ref, axis=0))
    np.testing.assert_allclose(emb * signs, emb_ref, atol=5e-2)


def test_spectral_torch_matches_jax():
    """Test that PyTorch and JAX implementations produce similar results."""
    n_samples = 100
    graph = scipy.sparse.random(n_samples, n_samples, density=0.1, random_state=42)
    graph = (graph + graph.T) / 2

    rng = np.random.default_rng(42)
    X = rng.random((n_samples, 10))
    dim = 2

    emb_jax = spectral_layout_jax(X, graph, dim, random_state=42)
    emb_torch = spectral_layout_torch(X, graph, dim, random_state=42)

    # Align signs by checking dot product of each dimension
    signs = np.sign(np.sum(emb_jax * emb_torch, axis=0))
    np.testing.assert_allclose(emb_jax, emb_torch * signs, atol=1e-5)
