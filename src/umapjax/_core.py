"""
Umap implementation using jax, core implementation.

This code was adapted from umap-learn.
Copyright (c) 2017, Leland McInnes
Licensed under the BSD 3-Clause License.
https://github.com/lmcinnes/umap/blob/master/LICENSE.txt
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import scipy.sparse as sp
from jaxtyping import Float
from sklearn.neighbors import KDTree
from umap import UMAP
from umap.spectral import spectral_layout as spectral_layout_scipy

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


ArrayLike = np.ndarray | sp.spmatrix
RandomState = int | np.random.RandomState | None

# Backend configuration: (module_path, function_name, install_extra)
_LAYOUT_BACKENDS = {
    "jax": ("umapjax.layouts.jax", "optimize_layout_euclidean", "jax"),
    "mx": ("umapjax.layouts.mlx", "optimize_layout_euclidean", "mlx"),
    "torch": ("umapjax.layouts.torch", "optimize_layout_euclidean", "torch"),
}

_SPECTRAL_BACKENDS = {
    "jax": ("umapjax.spectral.jax", "spectral_layout", "jax"),
    "torch": ("umapjax.spectral.torch", "spectral_layout", "torch"),
    "scipy": None,  # Uses scipy (always available)
}


def _lazy_import(backend: str, backends: dict, backend_type: str) -> Callable:
    """Lazily import a backend function."""
    if backend not in backends:
        raise ValueError(f"Unknown {backend_type}: {backend}")

    config = backends[backend]
    if config is None:
        return spectral_layout_scipy

    module_path, func_name, extra = config
    try:
        from importlib import import_module

        module = import_module(module_path)
        return getattr(module, func_name)
    except ImportError as e:
        raise ImportError(
            f"{backend} backend requires installation. Install with: pip install 'umapjax[{extra}]'"
        ) from e


def _get_layout_fn(backend: str) -> Callable:
    """Get the layout optimization function for the specified backend."""
    return _lazy_import(backend, _LAYOUT_BACKENDS, "layout_backend")


def _get_spectral_fn(backend: str) -> Callable:
    """Get the spectral layout function for the specified backend."""
    return _lazy_import(backend, _SPECTRAL_BACKENDS, "spectral_backend")


def simplicial_set_embedding(
    data: Float[np.ndarray, " n_samples n_features"] | sp.spmatrix,
    graph: sp.spmatrix,
    n_components: int,
    initial_alpha: float,
    a: float,
    b: float,
    gamma: float,
    negative_sample_rate: int,
    n_epochs: int | None,
    init: Literal["spectral", "random"] | Float[np.ndarray, " n_samples n_components"],
    random_state: RandomState,
    metric: str | Callable,
    metric_kwds: dict | None,
    verbose: bool = False,
    tqdm_kwds: dict | None = None,
    batch_size: int | None = None,
    spectral_backend: Literal["jax", "torch", "scipy"] = "scipy",
    layout_backend: Literal["jax", "mx", "torch"] = "jax",
) -> tuple[Float[np.ndarray, " n_samples n_components"], dict]:
    """Perform a fuzzy simplicial set embedding.

    Using a specified initialisation method and then minimizing the fuzzy set cross entropy
    between the 1-skeletons of the high and low dimensional fuzzy simplicial
    sets.

    Parameters
    ----------
    data
        The source data to be embedded by UMAP.

    graph
        The 1-skeleton of the high dimensional fuzzy simplicial set as
        represented by a graph for which we require a sparse matrix for the
        (weighted) adjacency matrix.

    n_components
        The dimensionality of the euclidean space into which to embed the data.

    initial_alpha
        Initial learning rate for the SGD.

    a
        Parameter of differentiable approximation of right adjoint functor

    b
        Parameter of differentiable approximation of right adjoint functor

    gamma
        Weight to apply to negative samples.

    negative_sample_rate
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.

    n_epochs
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If 0 is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).

    init
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.

    random_state
        A state capable being used as a numpy random state.

    metric
        The metric used to measure distance in high dimensional space; used if
        multiple connected components need to be layed out.

    metric_kwds
        Keyword arguments to be passed to the metric function; used if
        multiple connected components need to be layed out.

    verbose
        Whether to report information on the current progress of the algorithm.

    tqdm_kwds
        Key word arguments to be used by the tqdm progress bar.

    batch_size
        The batch size to use for the optimization loop.

    spectral_backend
        The backend to use for spectral layout.

    layout_backend
        The backend to use for the optimization loop.

    Returns
    -------
    embedding
        The optimized of ``graph`` into an ``n_components`` dimensional
        euclidean space.

    aux_data
        Empty dict for compatibility.
    """
    graph = graph.tocoo()
    graph.sum_duplicates()
    graph.shape[1]

    # For smaller datasets we can use more epochs
    if graph.shape[0] <= 10000:
        default_epochs = 500
    else:
        default_epochs = 200

    if n_epochs is None:
        n_epochs = default_epochs

    if n_epochs > 10:
        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    else:
        graph.data[graph.data < (graph.data.max() / float(default_epochs))] = 0.0

    graph.eliminate_zeros()

    if isinstance(init, str) and init == "random":
        embedding = random_state.uniform(low=-10.0, high=10.0, size=(graph.shape[0], n_components)).astype(np.float32)
    elif isinstance(init, str) and init == "spectral":
        # We add a little noise to avoid local minima for optimization to come
        spectral_fn = _get_spectral_fn(spectral_backend)
        initialisation = spectral_fn(
            data,
            graph,
            n_components,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )
        expansion = 10.0 / np.abs(initialisation).max()
        embedding = (initialisation * expansion).astype(np.float32) + random_state.normal(
            scale=0.0001, size=[graph.shape[0], n_components]
        ).astype(np.float32)
    else:
        init_data = np.array(init)
        if len(init_data.shape) == 2:
            if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
                tree = KDTree(init_data)
                dist, ind = tree.query(init_data, k=2)
                nndist = np.mean(dist[:, 1])
                embedding = init_data + random_state.normal(scale=0.001 * nndist, size=init_data.shape).astype(
                    np.float32
                )
            else:
                embedding = init_data

    head = graph.row
    tail = graph.col
    weight = graph.data

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    embedding = (10.0 * (embedding - np.min(embedding, 0)) / (np.max(embedding, 0) - np.min(embedding, 0))).astype(
        np.float32, order="C"
    )

    optimize_layout_fn = _get_layout_fn(layout_backend)

    embedding = optimize_layout_fn(
        head_embedding=embedding,
        tail_embedding=embedding,
        head=head,
        tail=tail,
        weight=weight,
        n_epochs=n_epochs,
        a=a,
        b=b,
        rng_state=rng_state,
        gamma=gamma,
        initial_alpha=initial_alpha,
        negative_sample_rate=negative_sample_rate,
        verbose=verbose,
        tqdm_kwds=tqdm_kwds,
        batch_size=batch_size,
    )
    return embedding, {}


class UmapJax(UMAP):
    """UMAP implementation using jax for acceleration.

    Finds a low dimensional embedding of the data that approximates
    an underlying manifold.

    Parameters
    ----------
    n_neighbors
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.

    n_components
        The dimension of the space to embed into. This defaults to 2 to
        provide easy visualization, but can reasonably be set to any
        integer value in the range 2 to 100.

    metric
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:

        * euclidean
        * manhattan
        * chebyshev
        * minkowski
        * canberra
        * braycurtis
        * mahalanobis
        * wminkowski
        * seuclidean
        * cosine
        * correlation
        * haversine
        * hamming
        * jaccard
        * dice
        * russelrao
        * kulsinski
        * ll_dirichlet
        * hellinger
        * rogerstanimoto
        * sokalmichener
        * sokalsneath
        * yule

        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.

    n_epochs
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).

    learning_rate
        The initial learning rate for the embedding optimization.

    init
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.

    spectral_backend
        The backend to use for the spectral embedding.

    min_dist
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.

    spread
        The effective scale of embedded points. In combination with ``min_dist``
        this determines how clustered/clumped the embedded points are.

    low_memory
        For some datasets the nearest neighbor computation can consume a lot of
        memory. If you find that UMAP is failing due to memory constraints
        consider setting this option to True. This approach is more
        computationally expensive, but avoids excessive memory use.

    set_op_mix_ratio
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.

    local_connectivity
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    repulsion_strength
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.

    negative_sample_rate
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.

    transform_queue_size
        For transform operations (embedding new points using a trained model
        this will control how aggressively to search for nearest neighbors.
        Larger values will result in slower performance but more accurate
        nearest neighbor evaluation.

    a
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    b
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.

    random_state
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    metric_kwds
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance. If None then no arguments are passed on.

    angular_rp_forest
        Whether to use an angular random projection forest to initialise
        the approximate nearest neighbor search. This can be faster, but is
        mostly on useful for metric that use an angular style distance such
        as cosine, correlation etc. In the case of those metrics angular forests
        will be chosen automatically.

    target_n_neighbors
        The number of nearest neighbors to use to construct the target simplcial
        set. If set to -1 use the ``n_neighbors`` value.

    target_metric
        The metric used to measure distance for a target array is using supervised
        dimension reduction. By default this is 'categorical' which will measure
        distance in terms of whether categories match or are different. Furthermore,
        if semi-supervised is required target values of -1 will be trated as
        unlabelled under the 'categorical' metric. If the target array takes
        continuous values (e.g. for a regression problem) then metric of 'l1'
        or 'l2' is probably more appropriate.

    target_metric_kwds
        Keyword argument to pass to the target metric when performing
        supervised dimension reduction. If None then no arguments are passed on.

    target_weight
        weighting factor between data topology and target topology. A value of
        0.0 weights predominantly on data, a value of 1.0 places a strong emphasis on
        target. The default of 0.5 balances the weighting equally between data and
        target.

    transform_seed
        Random seed used for the stochastic aspects of the transform operation.
        This ensures consistency in transform operations.

    verbose
        Controls verbosity of logging.

    tqdm_kwds
        Key word arguments to be used by the tqdm progress bar.

    unique
        Controls if the rows of your data should be uniqued before being
        embedded.  If you have more duplicates than you have n_neighbour
        you can have the identical data points lying in different regions of
        your space.  It also violates the definition of a metric.
        For to map from internal structures back to your data use the variable
        _unique_inverse_.

    densmap
        Specifies whether the density-augmented objective of densMAP
        should be used for optimization. Turning on this option generates
        an embedding where the local densities are encouraged to be correlated
        with those in the original space. Parameters below with the prefix 'dens'
        further control the behavior of this extension.

    dens_lambda
        Controls the regularization weight of the density correlation term
        in densMAP. Higher values prioritize density preservation over the
        UMAP objective, and vice versa for values closer to zero. Setting this
        parameter to zero is equivalent to running the original UMAP algorithm.

    dens_frac
        Controls the fraction of epochs (between 0 and 1) where the
        density-augmented objective is used in densMAP. The first
        (1 - dens_frac) fraction of epochs optimize the original UMAP objective
        before introducing the density correlation term.

    dens_var_shift
        A small constant added to the variance of local radii in the
        embedding when calculating the density correlation objective to
        prevent numerical instability from dividing by a small number

    output_dens
        Determines whether the local radii of the final embedding (an inverse
        measure of local density) are computed and returned in addition to
        the embedding. If set to True, local radii of the original data
        are also included in the output for comparison; the output is a tuple
        (embedding, original local radii, embedding local radii). This option
        can also be used when densmap=False to calculate the densities for
        UMAP embeddings.

    disconnection_distance
        Disconnect any vertices of distance greater than or equal to disconnection_distance when approximating the
        manifold via our k-nn graph. This is particularly useful in the case that you have a bounded metric.  The
        UMAP assumption that we have a connected manifold can be problematic when you have points that are maximally
        different from all the rest of your data.  The connected manifold assumption will make such points have perfect
        similarity to a random set of other points.  Too many such points will artificially connect your space.

    precomputed_knn
        If the k-nearest neighbors of each point has already been calculated you
        can pass them in here to save computation time. The number of nearest
        neighbors in the precomputed_knn must be greater or equal to the
        n_neighbors parameter. This should be a tuple containing the output
        of the nearest_neighbors() function or attributes from a previously fit
        UMAP object; (knn_indices, knn_dists,knn_search_index).

    batch_size
        The batch size to use for the optimization loop. If None, the batch size
        is set to the minimum of 8192 and the expected length of an epoch under standard UMAP sampling.

    layout_backend
        The backend to use for the optimization loop.
    """

    def __init__(
        self,
        n_neighbors: float = 15,
        n_components: int = 2,
        metric: str | Callable = "euclidean",
        metric_kwds: dict | None = None,
        output_metric: str | Callable = "euclidean",
        output_metric_kwds: dict | None = None,
        n_epochs: int | None = None,
        learning_rate: float = 1.0,
        init: Literal["spectral", "random"] | Float[np.ndarray, " n_samples n_components"] = "spectral",
        spectral_backend: Literal["jax", "torch", "scipy"] = "scipy",
        min_dist: float = 0.1,
        spread: float = 1.0,
        low_memory: bool = True,
        n_jobs: int = -1,
        set_op_mix_ratio: float = 1.0,
        local_connectivity: float = 1.0,
        repulsion_strength: float = 1.0,
        negative_sample_rate: int = 5,
        transform_queue_size: float = 4.0,
        a: float | None = None,
        b: float | None = None,
        random_state: RandomState = None,
        angular_rp_forest: bool = False,
        target_n_neighbors: int = -1,
        target_metric: str | Callable = "categorical",
        target_metric_kwds: dict | None = None,
        target_weight: float = 0.5,
        transform_seed: int = 42,
        transform_mode: str = "embedding",
        force_approximation_algorithm: bool = False,
        verbose: bool = False,
        tqdm_kwds: dict | None = None,
        unique: bool = False,
        densmap: bool = False,
        dens_lambda: float = 2.0,
        dens_frac: float = 0.3,
        dens_var_shift: float = 0.1,
        output_dens: bool = False,
        disconnection_distance: float | None = None,
        precomputed_knn: tuple[ArrayLike | None, ArrayLike | None, ArrayLike | None] = (None, None, None),
        batch_size: int | None = None,
        layout_backend: Literal["jax", "mx", "torch"] = "jax",
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.output_metric = output_metric
        self.target_metric = target_metric
        self.metric_kwds = metric_kwds
        self.output_metric_kwds = output_metric_kwds
        self.n_epochs = n_epochs
        self.init = init
        self.n_components = n_components
        self.repulsion_strength = repulsion_strength
        self.learning_rate = learning_rate
        self.spectral_backend = spectral_backend

        self.spread = spread
        self.min_dist = min_dist
        self.low_memory = low_memory
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.transform_queue_size = transform_queue_size
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_metric_kwds = target_metric_kwds
        self.target_weight = target_weight
        self.transform_seed = transform_seed
        self.transform_mode = transform_mode
        self.force_approximation_algorithm = force_approximation_algorithm
        self.verbose = verbose
        self.tqdm_kwds = tqdm_kwds
        self.unique = unique

        self.densmap = densmap
        self.dens_lambda = dens_lambda
        self.dens_frac = dens_frac
        self.dens_var_shift = dens_var_shift
        self.output_dens = output_dens
        self.disconnection_distance = disconnection_distance
        self.precomputed_knn = precomputed_knn
        self.batch_size = batch_size

        self.n_jobs = n_jobs
        self.layout_backend = layout_backend

        self.a = a
        self.b = b

        if self.output_metric != "euclidean":
            raise NotImplementedError("Only euclidean output metric is implemented.")
        if self.densmap:
            raise NotImplementedError("Densmap is not implemented.")

    def _fit_embed_data(self, X: ArrayLike, n_epochs: int, init: str, random_state: RandomState):
        """A method wrapper for simplicial_set_embedding that can be replaced by subclasses."""
        return simplicial_set_embedding(
            data=X,
            graph=self.graph_,
            n_components=self.n_components,
            initial_alpha=self._initial_alpha,
            a=self._a,
            b=self._b,
            gamma=self.repulsion_strength,
            negative_sample_rate=self.negative_sample_rate,
            n_epochs=n_epochs,
            init=init,
            spectral_backend=self.spectral_backend,
            random_state=random_state,
            metric=self._input_distance_func,
            metric_kwds=self._metric_kwds,
            verbose=self.verbose,
            tqdm_kwds=self.tqdm_kwds,
            batch_size=self.batch_size,
            layout_backend=self.layout_backend,
        )

    def transform(self, X: ArrayLike):
        """Not implemented."""
        raise NotImplementedError

    def inverse_transform(self, X: ArrayLike):
        """Not implemented."""
        raise NotImplementedError

    def update(self, X: ArrayLike):
        """Not implemented."""
        raise NotImplementedError

    def __add__(self):
        """Not implemented."""
        return NotImplementedError

    def __sub__(self):
        """Not implemented."""
        return NotImplementedError

    def __mul__(self):
        """Not implemented."""
        return NotImplementedError
