"""
Umap implementation using jax, optimization algorithms.

This code was adapted from umap-learn.
Copyright (c) 2017, Leland McInnes
Licensed under the BSD 3-Clause License.
https://github.com/lmcinnes/umap/blob/master/LICENSE.txt
"""

from __future__ import annotations

from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


@chex.dataclass
class OptimizationState:
    """State of the optimization."""

    head_embedding: Float[jnp.ndarray, " n_samples n_components"]
    key: Array
    alpha: float


@jax.jit
def rdist(x: Float[jnp.ndarray, " n_components"], y: Float[jnp.ndarray, " n_components"]) -> Float[jnp.ndarray, ""]:
    """Reduced Euclidean distance.

    Parameters
    ----------
    x
    y

    Returns
    -------
    The squared euclidean distance between x and y
    """
    diff = x - y
    return jnp.dot(diff, diff)


@partial(
    jax.jit,
    static_argnames=("gamma", "negative_sample_rate", "a", "b", "number_to_update", "move_other"),
    donate_argnums=(0, 1),
)
def _epoch_update(
    head_embedding: Float[jnp.ndarray, " n_samples n_components"],
    head_indices: Int[jnp.ndarray, " n_1_simplices"],
    tail_indices: Int[jnp.ndarray, " n_1_simplices"],
    edge_weights: Float[jnp.ndarray, " n_1_simplices"],
    j_s: Int[jnp.ndarray, " number_to_update negative_sample_rate"],
    alpha: float,
    negative_sample_rate: int,
    gamma: float,
    a: float,
    b: float,
    number_to_update: int,
    move_other: bool,
) -> Float[jnp.ndarray, " n_samples n_components"]:
    """Update the attractive forces between vertices in the embedding.

    Parameters
    ----------
    head_embedding
        The embedding of the query sample.
    head_indices
        The indices of the query samples.
    tail_indices
        The indices of the reference samples.
    edge_weights
        The weight of the 1-simplices.
    j_s
        Indices of negative samples.
    alpha
        The learning rate.
    negative_sample_rate
        The number of negative samples to use for each positive sample.
    gamma
        The repulsion strength.
    a
        Parameter of differentiable approximation of right adjoint functor
    b
        Parameter of differentiable approximation of right adjoint functor
    number_to_update
        The number of samples to update.
    move_other
        Whether to move the tail samples in the 1-simplices.
    """

    def _attractive_grad_coeff(dist_squared):
        grad_coeff = -2 * a * b * jnp.power(dist_squared, b - 1.0)
        grad_coeff /= a * jnp.power(dist_squared, b) + 1.0
        return jnp.nan_to_num(grad_coeff, nan=0.0)

    def _repulsive_grad_coeff(dist_squared):
        grad_coeff = 2 * gamma * b
        grad_coeff /= (0.001 + dist_squared) * (a * jnp.power(dist_squared, b) + 1)
        return jnp.nan_to_num(grad_coeff, nan=0.0)

    tail_embedding = head_embedding
    current = head_embedding[head_indices]
    other = tail_embedding[tail_indices]
    dist_squared = jax.vmap(rdist)(current, other)[:, None]
    grad_coeff = _attractive_grad_coeff(dist_squared)
    grad = jnp.clip(grad_coeff * (current - other), -4.0, 4.0)

    tail_embedding_rep = tail_embedding[j_s]
    dist_squared_rep = jax.vmap(jax.vmap(rdist, in_axes=(None, 0)))(current, tail_embedding_rep)
    chex.assert_shape(dist_squared_rep, (number_to_update, negative_sample_rate))

    grad_coeff_rep = _repulsive_grad_coeff(dist_squared_rep)
    grad_rep = grad_coeff_rep[:, :, None] * (current[:, None, :] - tail_embedding_rep)
    # Removes any self-interaction gradients
    grad_rep = jnp.where(dist_squared_rep[:, :, None] == 0.0, 0.0, grad_rep)
    grad_rep = jnp.where(grad_coeff_rep[:, :, None] > 0.0, grad_rep, 0.0)
    grad_rep = jnp.clip(grad_rep, -4.0, 4.0)
    grad_rep = jnp.sum(grad_rep, axis=1)

    grad_head = grad + grad_rep
    grad_tail = -grad if move_other else jnp.zeros_like(grad)
    scale_grad = lambda g: alpha * g * edge_weights[:, None]

    return scale_grad(grad_head), scale_grad(grad_tail)


def optimize_layout_euclidean(
    head_embedding: Float[jnp.ndarray, " n_samples n_components"],
    tail_embedding: Float[jnp.ndarray, " source_samples n_components"],
    head: Int[jnp.ndarray, " n_1_simplices"],
    tail: Int[jnp.ndarray, " n_1_simplices"],
    weight: Float[jnp.ndarray, " n_1_simplices"],
    n_epochs: int,
    a: float,
    b: float,
    rng_state: Int[np.ndarray, " n_keys"],
    gamma: float = 1.0,
    initial_alpha: float = 1.0,
    negative_sample_rate: float = 5.0,
    verbose: bool = False,
    tqdm_kwds: dict | None = None,
    batch_size: int | None = None,
) -> Float[np.ndarray, " n_samples n_components"]:
    """Perform the optimization.

    Improve an embedding using gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. This implementation uses
    weighted gradients rather than stochastic sampling.

    Parameters
    ----------
    head_embedding
        The initial embedding to be improved by SGD.
    tail_embedding
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.
    head
        The indices of the heads of 1-simplices with non-zero membership.
    tail
        The indices of the tails of 1-simplices with non-zero membership.
    weight
        The membership strength of each 1-simplex.
    n_epochs
        The number of training epochs to use in optimization.
    n_vertices: int
        The number of vertices (0-simplices) in the dataset.
    a
        Parameter of differentiable approximation of right adjoint functor
    b
        Parameter of differentiable approximation of right adjoint functor
    rng_state
        The internal state of the rng
    gamma
        Weight to apply to negative samples.
    initial_alpha
        Initial learning rate for the SGD.
    negative_sample_rate
        Number of negative samples to use per positive sample.
    verbose
        Whether to report information on the current progress of the algorithm.
    tqdm_kwds
        Keyword arguments for tqdm progress bar.
    batch_size
        The batch size to use for the optimization loop.

    Returns
    -------
    embedding
        The optimized embedding.
    """
    n_1_simplices = weight.shape[0]
    n_epochs = float(n_epochs)

    if tqdm_kwds is None:
        tqdm_kwds = {}
    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    chex.assert_trees_all_equal(head_embedding, tail_embedding)
    head_embedding = jnp.asarray(head_embedding)
    head = jnp.asarray(head)
    tail = jnp.asarray(tail)

    # Normalize weights
    weight = jnp.asarray(weight / np.max(weight), dtype=jnp.float32)
    # Clip so that we have a total weight of 1 within n_epochs for the least weighted edge.
    # This is not implemented in umap-learn
    weight = jnp.clip(weight, 1 / n_epochs, 1.0)

    batch_size = batch_size or min(head_embedding.shape[0], int(n_1_simplices))
    negative_sample_rate = int(negative_sample_rate)

    # Pad arrays to be divisible by batch_size
    n_batches = int(np.ceil(n_1_simplices / batch_size))
    padding_len = n_batches * batch_size - n_1_simplices

    def _epoch(n, state):
        alpha = state.alpha
        key, subkey = jax.random.split(state.key)

        # Shuffle indices
        indices = jnp.arange(n_1_simplices)
        if padding_len > 0:
            indices = jnp.concatenate([indices, jnp.full((padding_len,), -1, dtype=jnp.int32)])
        shuffled_indices = jax.random.permutation(key, indices)
        batched_indices = shuffled_indices.reshape((n_batches, batch_size))

        def _batch(carry, batch_indices):
            curr_head_embedding, key = carry

            key, subkey = jax.random.split(key)
            j_s = jax.random.randint(
                subkey,
                (batch_size, negative_sample_rate),
                0,
                curr_head_embedding.shape[0],
            )

            head_update, tail_update = _epoch_update(
                curr_head_embedding,
                head_indices=head[batch_indices],
                tail_indices=tail[batch_indices],
                edge_weights=weight[batch_indices],
                j_s=j_s,
                alpha=alpha,
                gamma=gamma,
                negative_sample_rate=negative_sample_rate,
                a=a,
                b=b,
                number_to_update=batch_size,
                move_other=True,
            )

            # Mask updates for padded samples (where indices are -1)
            head_update = jnp.where(batch_indices[:, None] != -1, head_update, 0.0)
            tail_update = jnp.where(batch_indices[:, None] != -1, tail_update, 0.0)
            curr_head_embedding = curr_head_embedding.at[head[batch_indices]].add(head_update)
            curr_head_embedding = curr_head_embedding.at[tail[batch_indices]].add(tail_update)

            return (curr_head_embedding, key), None

        (head_embedding, key), _ = jax.lax.scan(_batch, (state.head_embedding, key), batched_indices)

        return state.replace(
            alpha=initial_alpha * (1.0 - (n / n_epochs)),
            head_embedding=head_embedding,
            key=key,
        )

    state = OptimizationState(
        head_embedding=head_embedding,
        key=jax.random.PRNGKey(rng_state[0]),
        alpha=initial_alpha,
    )
    state = jax.lax.fori_loop(0, int(n_epochs), _epoch, state)
    return np.array(jax.device_get(state.head_embedding))
