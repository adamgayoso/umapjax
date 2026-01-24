# umapjax

[![Tests][badge-tests]][tests]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/adamgayoso/umapjax/test.yaml?branch=main

UMAP, but optimized with jax. (Experimental implementation)

`umapjax` inherits the API of [umap-learn](https://umap-learn.readthedocs.io/en/latest/). The `UmapJax` class is a drop-in replacement for `umap.UMAP`, with a few key differences:

1. `umapjax` does not support `densmap`.
2. `umapjax` does not support `output_metric` other than `euclidean`.

**Note:** `umapjax` does not fully replicate `umap-learn` and care should be used when interpreting results.

This package is intended to be used in combination with accelerated hardware like GPUs and TPUs. There is no benefit to using `umapjax` on a CPU.

## Getting started

```python
import umapjax

model = umapjax.UmapJax(n_neighbors=15)
embedding = model.fit_transform(X)
```

## Implementation details

The implementaion used in `umapjax` is very similar to the one used in [umap-learn](https://umap-learn.readthedocs.io/en/latest/); however, rather than a single step updating one single point, we update a set of points in parallel using jax. The gradients of the points are weighted by edge weights, which control sampling frequencies in the original algorithm. If results look strange, try changing `n_epochs` or `batch_size`. The `batch_size` argument can also be used to control acceleration on GPUs/TPUs.

## Installation

You need to have Python 3.11 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install umapjax:

1. Install the latest release of `umapjax` from [PyPI][]:

```bash
pip install umapjax
```

2. Install the latest development version:

```bash
pip install git+https://github.com/adamgayoso/umapjax.git@main
```

## Release notes

See the [changelog][].

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
