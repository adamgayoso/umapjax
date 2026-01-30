# umapjax

[![Tests][badge-tests]][tests]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/adamgayoso/umapjax/test.yaml?branch=main

UMAP, but accelerated. (Experimental implementation)

`umapjax` inherits the API of [umap-learn](https://umap-learn.readthedocs.io/en/latest/). The `UmapJax` class is a drop-in replacement for `umap.UMAP`, with a few key differences:

1. `umapjax` does not support `densmap`.
2. `umapjax` does not support `output_metric` other than `euclidean`.

**Note:** `umapjax` does not fully replicate `umap-learn` and care should be used when interpreting results.

This package implements the following backends (despite being named `umapjax`):

1. `torch` (PyTorch)
2. `mx` (MLX)
3. `jax` (JAX)

## Getting started

```python
import umapjax

layout_backend: Literal["jax", "mx", "torch"] = "jax"
spectral_backend: Literal["jax", "scipy", "torch"] = "scipy"
batch_size: int | None = None # Defaults to X.shape[0]

model = umapjax.UmapJax(
    n_neighbors=15,
    layout_backend=layout_backend,
    spectral_backend=spectral_backend
)
embedding = model.fit_transform(X)
```

If the optimization is slow, try increasing the batch size as a multiple of `X.shape[0]`. All backends will automatically use accelerated hardware if available.

If using `"torch"`, you can set `umapjax.layouts.torch.TORCH_DEVICE` and `umapjax.spectral.torch.TORCH_DEVICE` to control the default device used for the layout and spectral embedding, respectively.

## Implementation details

The implementaion used in `umapjax` is very similar to the one used in [umap-learn](https://umap-learn.readthedocs.io/en/latest/); however, rather than a single step updating one single point, we update a set of points in parallel using jax. The gradients of the points are weighted by edge weights, which control sampling frequencies in the original algorithm. If results look strange, try changing `n_epochs` or `batch_size`. The `batch_size` argument can also be used to control acceleration on GPUs/TPUs.

## Installation

You need to have Python 3.11 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install umapjax:

1. Install the latest release of `umapjax` from [PyPI][] with a preferred backend:

```bash
pip install "umapjax[jax,mlx,torch]"
```

2. Install the latest development version:

```bash
pip install "umapjax[jax,mlx,torch] @ git+https://github.com/adamgayoso/umapjax.git@main"
```

## Contact

If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/adamgayoso/umapjax/issues
[tests]: https://github.com/adamgayoso/umapjax/actions/workflows/test.yaml
[documentation]: https://umapjax.readthedocs.io
[changelog]: https://umapjax.readthedocs.io/en/latest/changelog.html
[api documentation]: https://umapjax.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/umapjax
