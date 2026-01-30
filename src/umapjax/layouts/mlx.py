"""
Umap implementation using MLX, optimization algorithms.

This code was adapted from umap-learn.
Copyright (c) 2017, Leland McInnes
Licensed under the BSD 3-Clause License.
https://github.com/lmcinnes/umap/blob/master/LICENSE.txt
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class OptimizationState:
    """State for the UMAP optimization loop."""

    head_embedding: mx.array
    key: mx.array
    alpha: float


def _attractive_grad_coeff(dist_squared: mx.array, a: float, b: float) -> mx.array:
    """Compute the attractive gradient coefficient."""
    a_mx = mx.array(a)
    b_mx = mx.array(b)
    grad_coeff = -2 * a_mx * b_mx * mx.power(dist_squared, b_mx - 1.0)
    grad_coeff = grad_coeff / (a_mx * mx.power(dist_squared, b_mx) + 1.0)
    return mx.where(mx.isfinite(grad_coeff), grad_coeff, 0.0)


def _repulsive_grad_coeff(dist_squared: mx.array, gamma: float, a: float, b: float) -> mx.array:
    """Compute the repulsive gradient coefficient."""
    gamma_mx = mx.array(gamma)
    a_mx = mx.array(a)
    b_mx = mx.array(b)
    grad_coeff = 2 * gamma_mx * b_mx
    grad_coeff = grad_coeff / ((0.001 + dist_squared) * (a_mx * mx.power(dist_squared, b_mx) + 1))
    return mx.where(mx.isfinite(grad_coeff), grad_coeff, 0.0)


def _epoch_update(
    head_embedding: mx.array,
    head_indices: mx.array,
    tail_indices: mx.array,
    edge_weights: mx.array,
    j_s: mx.array,
    alpha: mx.array,
    negative_sample_rate: int,
    gamma: float,
    a: float,
    b: float,
    move_other: bool,
) -> tuple[mx.array, mx.array]:
    """Pure gradient computation function (mx.compile compatible).

    Parameters
    ----------
    head_embedding
        The current embedding (n_samples, n_components)
    head_indices
        Indices of head samples (batch_size,)
    tail_indices
        Indices of tail samples (batch_size,)
    edge_weights
        The weight of the 1-simplices (batch_size,)
    j_s
        Indices of negative samples (batch_size, negative_sample_rate)
    alpha
        The learning rate.
    negative_sample_rate
        Number of negative samples.
    gamma
        The repulsion strength.
    a
        Parameter of differentiable approximation of right adjoint functor
    b
        Parameter of differentiable approximation of right adjoint functor
    move_other
        Whether to update tail embeddings.

    Returns
    -------
    grad_head, grad_tail
        Gradient updates for head and tail embeddings
    """
    # Slice embeddings
    current = head_embedding[head_indices]
    other = head_embedding[tail_indices]

    # Compute attractive forces using broadcasting instead of vmap
    diff_attr = current - other
    dist_squared = mx.sum(diff_attr * diff_attr, axis=-1, keepdims=True)
    grad_coeff = _attractive_grad_coeff(dist_squared, a, b)
    grad = mx.clip(grad_coeff * diff_attr, -4.0, 4.0)

    # Compute repulsive forces
    tail_embedding_rep = head_embedding[j_s]
    # diff_rep: (batch_size, negative_sample_rate, n_components)
    diff_rep = current[:, None, :] - tail_embedding_rep
    # dist_squared_rep: (batch_size, negative_sample_rate)
    dist_squared_rep = mx.sum(diff_rep * diff_rep, axis=-1)

    grad_coeff_rep = _repulsive_grad_coeff(dist_squared_rep, gamma, a, b)
    grad_rep = grad_coeff_rep[:, :, None] * diff_rep

    # Removes any self-interaction gradients
    grad_rep = mx.where(dist_squared_rep[:, :, None] == 0.0, 0.0, grad_rep)
    grad_rep = mx.where(grad_coeff_rep[:, :, None] > 0.0, grad_rep, 0.0)
    grad_rep = mx.clip(grad_rep, -4.0, 4.0)
    grad_rep = mx.sum(grad_rep, axis=1)

    grad_head = grad + grad_rep

    if move_other:
        grad_tail = -grad
    else:
        grad_tail = mx.zeros_like(grad)

    # Scale gradients
    scale = alpha * edge_weights[:, None]
    return grad_head * scale, grad_tail * scale


def optimize_layout_euclidean(
    head_embedding: np.ndarray,
    tail_embedding: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    weight: np.ndarray,
    n_epochs: int,
    a: float,
    b: float,
    rng_state: np.ndarray,
    gamma: float = 1.0,
    initial_alpha: float = 1.0,
    negative_sample_rate: float = 5.0,
    verbose: bool = False,
    tqdm_kwds: dict | None = None,
    batch_size: int | None = None,
) -> np.ndarray:
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
    n_epochs_float = float(n_epochs)

    if tqdm_kwds is None:
        tqdm_kwds = {}
    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    # Check embeddings are the same (self-embedding case)
    if not np.array_equal(head_embedding, tail_embedding):
        raise ValueError("head_embedding and tail_embedding must be the same array for self-embedding")

    # Convert to MLX arrays
    head_embedding_mx = mx.array(head_embedding)
    head_mx = mx.array(head)
    tail_mx = mx.array(tail)

    # Normalize weights
    weight_mx = mx.array(weight / np.max(weight), dtype=mx.float32)
    # Clip so that we have a total weight of 1 within n_epochs for the least weighted edge.
    weight_mx = mx.clip(weight_mx, 1 / n_epochs_float, 1.0)

    batch_size = batch_size or min(head_embedding.shape[0], int(n_1_simplices))
    negative_sample_rate = int(negative_sample_rate)

    # Pad arrays to be divisible by batch_size
    n_batches = int(np.ceil(n_1_simplices / batch_size))
    padding_len = n_batches * batch_size - n_1_simplices

    # Compile the gradient computation function.
    # Note: We partially apply the scalar constants and static configurations.
    compiled_epoch_update = mx.compile(
        functools.partial(
            _epoch_update, a=a, b=b, gamma=gamma, negative_sample_rate=negative_sample_rate, move_other=True
        )
    )

    # Initialize state
    seed = abs(rng_state[0].item() if hasattr(rng_state[0], "item") else int(rng_state[0]))
    key = mx.random.key(seed)
    alpha = mx.array(initial_alpha)

    # Main training loop
    for epoch in range(int(n_epochs)):
        key, subkey = mx.random.split(key)

        # Shuffle indices
        indices = mx.arange(n_1_simplices)
        if padding_len > 0:
            indices = mx.concatenate([indices, mx.full((padding_len,), -1, dtype=mx.int32)])
        shuffled_indices = mx.random.permutation(indices, key=subkey)
        batched_indices = shuffled_indices.reshape((n_batches, batch_size))

        # Process each batch
        for batch_idx in range(n_batches):
            batch_indices = batched_indices[batch_idx]

            key, subkey = mx.random.split(key)
            j_s = mx.random.randint(
                low=0,
                high=head_embedding_mx.shape[0],
                shape=(batch_size, negative_sample_rate),
                key=subkey,
            )

            current_head_indices = head_mx[batch_indices]
            current_tail_indices = tail_mx[batch_indices]
            current_edge_weights = weight_mx[batch_indices]

            # Compute gradients
            head_update, tail_update = compiled_epoch_update(
                head_embedding=head_embedding_mx,
                head_indices=current_head_indices,
                tail_indices=current_tail_indices,
                edge_weights=current_edge_weights,
                j_s=j_s,
                alpha=alpha,
            )

            # Mask updates for padded samples (where indices are -1)
            head_update = mx.where(batch_indices[:, None] != -1, head_update, 0.0)
            tail_update = mx.where(batch_indices[:, None] != -1, tail_update, 0.0)

            # Apply updates
            head_embedding_mx[current_head_indices] += head_update
            head_embedding_mx[current_tail_indices] += tail_update

        # Update learning rate
        alpha = initial_alpha * (1.0 - (epoch / n_epochs_float))

        # Evaluate periodically to prevent graph from growing too large
        if epoch % 100 == 0:
            mx.eval(head_embedding_mx)

    # Final evaluation and convert back to numpy
    mx.eval(head_embedding_mx)
    return np.array(head_embedding_mx)
