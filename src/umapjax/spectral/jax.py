"""
Spectral embedding of the graph using jax.

This code was adapted from umap-learn.
Copyright (c) 2017, Leland McInnes
Licensed under the BSD 3-Clause License.
https://github.com/lmcinnes/umap/blob/master/LICENSE.txt
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import umap
from jax.experimental.sparse import bcoo, linalg
from jaxtyping import Float
from sklearn.metrics import pairwise_distances

RandomState = int | np.random.Generator | np.random.RandomState | None

SPECTRAL_DENSE_THRESHOLD = 10_000  # ~1.2GB of RAM in float32


def multi_component_layout(
    data: Float[np.ndarray, " n_samples n_features"] | scipy.sparse.spmatrix,
    graph: scipy.sparse.spmatrix,
    n_components: int,
    component_labels: np.ndarray,
    dim: int,
    random_state: RandomState,
    metric: str = "euclidean",
    metric_kwds: dict | None = None,
    tol: float = 0.0,
    maxiter: int = 0,
) -> Float[np.ndarray, " n_samples dim"]:
    """Specialised layout algorithm for dealing with graphs with many connected components.

    This will first find relative positions for the components by spectrally embedding
    their centroids, then spectrally embed each individual connected component positioning
    them according to the centroid embeddings.

    Parameters
    ----------
    data
        The source data -- required so we can generate centroids for each
        connected component of the graph.
    graph
        The adjacency matrix of the graph to be embedded.
    n_components
        The number of distinct components to be layed out.
    component_labels
        For each vertex in the graph the label of the component to
        which the vertex belongs.
    dim
        The chosen embedding dimension.
    random_state
        A state capable being used as a numpy random state.
    metric
        The metric used to measure distances among the source data points.
    metric_kwds
        Keyword arguments to be passed to the metric function.
    tol
        Stopping tolerance for the numerical algorithm computing the embedding.
    maxiter
        Number of iterations the numerical algorithm will go through at most as it
        attempts to compute the embedding.

    Returns
    -------
    embedding
        The initial embedding of ``graph``.
    """
    if metric_kwds is None:
        metric_kwds = {}
    result = np.empty((graph.shape[0], dim), dtype=np.float32)

    if n_components > 2 * dim:
        meta_embedding = umap.spectral.component_layout(
            data,
            n_components,
            component_labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )
    else:
        k = int(np.ceil(n_components / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta_embedding = np.vstack([base, -base])[:n_components]

    for label in range(n_components):
        component_graph = graph.tocsr()[component_labels == label, :].tocsc()
        component_graph = component_graph[:, component_labels == label].tocoo()

        distances = pairwise_distances([meta_embedding[label]], meta_embedding)
        data_range = distances[distances > 0.0].min() / 2.0

        if component_graph.shape[0] < 2 * dim or component_graph.shape[0] <= dim + 1:
            result[component_labels == label] = (
                random_state.uniform(
                    low=-data_range,
                    high=data_range,
                    size=(component_graph.shape[0], dim),
                )
                + meta_embedding[label]
            )
        else:
            component_embedding = _spectral_layout(
                data=None,
                graph=component_graph,
                dim=dim,
                random_state=random_state,
                metric=metric,
                metric_kwds=metric_kwds,
                tol=tol,
                maxiter=maxiter,
            )
            expansion = data_range / np.max(np.abs(component_embedding))
            component_embedding *= expansion
            result[component_labels == label] = component_embedding + meta_embedding[label]

    return result


def spectral_layout(
    data: Float[np.ndarray, " n_samples n_features"] | scipy.sparse.spmatrix,
    graph: scipy.sparse.spmatrix,
    dim: int,
    random_state: RandomState,
    metric: str = "euclidean",
    metric_kwds: dict | None = None,
    tol: float = 0.0,
    maxiter: int = 0,
) -> Float[np.ndarray, " n_samples dim"]:
    """Compute spectral layout with jax.

    Parameters
    ----------
    data
        The source data

    graph
        The (weighted) adjacency matrix of the graph as a sparse matrix.

    dim
        The dimension of the space into which to embed.

    random_state
        A state capable being used as a numpy random state.

    tol
        Stopping tolerance for the numerical algorithm computing the embedding.

    maxiter
        Number of iterations the numerical algorithm will go through at most as it
        attempts to compute the embedding.

    Returns
    -------
    embedding
        The spectral embedding of the graph.
    """
    if metric_kwds is None:
        metric_kwds = {}
    return _spectral_layout(
        data=data,
        graph=graph,
        dim=dim,
        random_state=random_state,
        metric=metric,
        metric_kwds=metric_kwds,
        tol=tol,
        maxiter=maxiter,
    )


def _spectral_layout(
    data: Float[np.ndarray, " n_samples n_features"] | scipy.sparse.spmatrix,
    graph: scipy.sparse.spmatrix,
    dim: int,
    random_state: RandomState,
    metric: str = "euclidean",
    metric_kwds: dict | None = None,
    tol: float = 0.0,
    maxiter: int = 0,
) -> Float[np.ndarray, " n_samples dim"]:
    """Spectral embedding of the graph using jax.

    Parameters
    ----------
    data
        The source data
    graph
        The (weighted) adjacency matrix of the graph as a sparse matrix.
    dim
        The dimension of the space into which to embed.
    random_state
        A state capable being used as a numpy random state.
    metric
        The metric used to measure distances among the source data points.
        Used only if the multiple connected components are found in the
        graph.
    metric_kwds
        Keyword arguments to be passed to the metric function.
    tol
        Stopping tolerance for the numerical algorithm computing the embedding.
    maxiter
        Number of iterations the numerical algorithm will go through at most as it
        attempts to compute the embedding.

    Returns
    -------
    embedding
        The spectral embedding of the graph.
    """
    if metric_kwds is None:
        metric_kwds = {}
    n_components, labels = scipy.sparse.csgraph.connected_components(graph)

    if n_components > 1:
        return multi_component_layout(
            data,
            graph,
            n_components,
            labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )

    sqrt_deg = np.sqrt(np.asarray(graph.sum(axis=0)).squeeze())
    # Determine dtype based on configuration (float32 or float64)
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32

    I = scipy.sparse.identity(graph.shape[0], dtype=dtype)
    D = scipy.sparse.spdiags(1.0 / sqrt_deg, 0, graph.shape[0], graph.shape[0])
    L = I - D * graph * D
    if not scipy.sparse.issparse(L):
        L = np.asarray(L)

    k = dim + 1
    gen = (
        random_state
        if isinstance(random_state, (np.random.Generator, np.random.RandomState))
        else np.random.default_rng(seed=random_state)
    )
    X = gen.normal(size=(L.shape[0], k)).astype(dtype)

    if graph.shape[0] < SPECTRAL_DENSE_THRESHOLD:
        L_dense = L.toarray() if scipy.sparse.issparse(L) else L
        eigenvalues, eigenvectors = jnp.linalg.eigh(jnp.asarray(L_dense))
        order = np.argsort(np.array(eigenvalues))[1:k]
        return np.array(jax.device_get(eigenvectors))[:, order]

    L_sparse = bcoo.BCOO.from_scipy_sparse(L)
    eigenvalues, eigenvectors, _ = linalg.lobpcg_standard(
        lambda v: -1 * L_sparse @ v,  # -1 trick to get smallest eigenvalues
        jnp.asarray(X),
        tol=tol or 1e-4,
        m=maxiter or 5 * graph.shape[0],
    )
    eigenvalues *= -1
    order = np.argsort(np.array(jax.device_get(eigenvalues)))[1:k]
    return np.array(jax.device_get(eigenvectors))[:, order]
