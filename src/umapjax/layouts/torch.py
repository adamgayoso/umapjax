"""
Umap implementation using PyTorch, optimization algorithms.

This code was adapted from umap-learn.
Copyright (c) 2017, Leland McInnes
Licensed under the BSD 3-Clause License.
https://github.com/lmcinnes/umap/blob/master/LICENSE.txt
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, Int

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class OptimizationState:
    """State for the UMAP optimization loop."""

    head_embedding: Float[torch.Tensor, " n_samples n_components"]
    generator: torch.Generator
    alpha: float


def _attractive_grad_coeff(dist_squared: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Compute the attractive gradient coefficient."""
    grad_coeff = -2 * a * b * torch.pow(dist_squared, b - 1.0)
    grad_coeff = grad_coeff / (a * torch.pow(dist_squared, b) + 1.0)
    return torch.nan_to_num(grad_coeff, nan=0.0)


def _repulsive_grad_coeff(dist_squared: torch.Tensor, gamma: float, a: float, b: float) -> torch.Tensor:
    """Compute the repulsive gradient coefficient."""
    grad_coeff = 2 * gamma * b
    grad_coeff = grad_coeff / ((0.001 + dist_squared) * (a * torch.pow(dist_squared, b) + 1))
    return torch.nan_to_num(grad_coeff, nan=0.0)


def _epoch_update(
    head_embedding: Float[torch.Tensor, " n_samples n_components"],
    head_indices: Int[torch.Tensor, " batch_size"],
    tail_indices: Int[torch.Tensor, " batch_size"],
    edge_weights: Float[torch.Tensor, " batch_size"],
    j_s: Int[torch.Tensor, " batch_size negative_sample_rate"],
    alpha: Float[torch.Tensor, ""],
    negative_sample_rate: int,
    gamma: float,
    a: float,
    b: float,
    move_other: bool,
) -> tuple[Float[torch.Tensor, " batch_size n_components"], Float[torch.Tensor, " batch_size n_components"]]:
    """Pure gradient computation function (torch.compile compatible).

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

    # Compute attractive forces using broadcasting
    diff_attr = current - other
    dist_squared = torch.sum(diff_attr * diff_attr, dim=-1, keepdim=True)
    grad_coeff = _attractive_grad_coeff(dist_squared, a, b)
    grad = torch.clamp(grad_coeff * diff_attr, -4.0, 4.0)

    # Compute repulsive forces
    tail_embedding_rep = head_embedding[j_s]
    # diff_rep: (batch_size, negative_sample_rate, n_components)
    diff_rep = current[:, None, :] - tail_embedding_rep
    # dist_squared_rep: (batch_size, negative_sample_rate)
    dist_squared_rep = torch.sum(diff_rep * diff_rep, dim=-1)

    grad_coeff_rep = _repulsive_grad_coeff(dist_squared_rep, gamma, a, b)
    grad_rep = grad_coeff_rep[:, :, None] * diff_rep

    # Removes any self-interaction gradients
    grad_rep = torch.where(dist_squared_rep[:, :, None] == 0.0, torch.zeros_like(grad_rep), grad_rep)
    grad_rep = torch.where(grad_coeff_rep[:, :, None] > 0.0, grad_rep, torch.zeros_like(grad_rep))
    grad_rep = torch.clamp(grad_rep, -4.0, 4.0)
    grad_rep = torch.sum(grad_rep, dim=1)

    grad_head = grad + grad_rep

    if move_other:
        grad_tail = -grad
    else:
        grad_tail = torch.zeros_like(grad)

    # Scale gradients
    scale = alpha * edge_weights[:, None]
    return grad_head * scale, grad_tail * scale


def optimize_layout_euclidean(
    head_embedding: Float[np.ndarray, " n_samples n_components"],
    tail_embedding: Float[np.ndarray, " source_samples n_components"],
    head: Int[np.ndarray, " n_1_simplices"],
    tail: Int[np.ndarray, " n_1_simplices"],
    weight: Float[np.ndarray, " n_1_simplices"],
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

    # Determine device
    device = TORCH_DEVICE

    # Convert to torch tensors
    head_embedding_t = torch.tensor(head_embedding, dtype=torch.float32, device=device)
    head_t = torch.tensor(head, dtype=torch.long, device=device)
    tail_t = torch.tensor(tail, dtype=torch.long, device=device)

    # Normalize weights
    weight_t = torch.tensor(weight / np.max(weight), dtype=torch.float32, device=device)
    # Clip so that we have a total weight of 1 within n_epochs for the least weighted edge.
    weight_t = torch.clamp(weight_t, 1 / n_epochs_float, 1.0)

    batch_size = batch_size or min(head_embedding.shape[0], int(n_1_simplices))
    negative_sample_rate = int(negative_sample_rate)

    # Pad arrays to be divisible by batch_size
    n_batches = int(np.ceil(n_1_simplices / batch_size))
    padding_len = n_batches * batch_size - n_1_simplices

    # Compile the gradient computation function.
    compiled_epoch_update = torch.compile(
        functools.partial(
            _epoch_update, a=a, b=b, gamma=gamma, negative_sample_rate=negative_sample_rate, move_other=True
        )
    )

    # Initialize random generator
    seed = abs(rng_state[0].item() if hasattr(rng_state[0], "item") else int(rng_state[0]))
    generator = torch.Generator(device=device).manual_seed(seed)
    alpha = torch.tensor(initial_alpha, dtype=torch.float32, device=device)

    # Main training loop
    for epoch in range(int(n_epochs)):
        # Shuffle indices
        indices = torch.arange(n_1_simplices, device=device)
        if padding_len > 0:
            padding = torch.full((padding_len,), -1, dtype=torch.long, device=device)
            indices = torch.cat([indices, padding])
        shuffled_indices = indices[torch.randperm(indices.shape[0], generator=generator, device=device)]
        batched_indices = shuffled_indices.reshape((n_batches, batch_size))

        # Process each batch
        for batch_idx in range(n_batches):
            batch_indices = batched_indices[batch_idx]

            j_s = torch.randint(
                low=0,
                high=head_embedding_t.shape[0],
                size=(batch_size, negative_sample_rate),
                generator=generator,
                device=device,
            )

            current_head_indices = head_t[batch_indices]
            current_tail_indices = tail_t[batch_indices]
            current_edge_weights = weight_t[batch_indices]

            # Compute gradients
            head_update, tail_update = compiled_epoch_update(
                head_embedding=head_embedding_t,
                head_indices=current_head_indices,
                tail_indices=current_tail_indices,
                edge_weights=current_edge_weights,
                j_s=j_s,
                alpha=alpha,
            )

            # Mask updates for padded samples (where indices are -1)
            mask = (batch_indices != -1).unsqueeze(-1)
            head_update = torch.where(mask, head_update, torch.zeros_like(head_update))
            tail_update = torch.where(mask, tail_update, torch.zeros_like(tail_update))

            # Apply updates using index_add for proper gradient accumulation
            head_embedding_t.index_add_(0, current_head_indices, head_update)
            head_embedding_t.index_add_(0, current_tail_indices, tail_update)

        # Update learning rate
        alpha = torch.tensor(initial_alpha * (1.0 - (epoch / n_epochs_float)), dtype=torch.float32, device=device)

    # Convert back to numpy
    return head_embedding_t.cpu().numpy()
