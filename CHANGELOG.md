# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.1.0]

### Added

- Multiple backends for layout and spectral initialization, improved API. See README for usage.

## [0.0.3]

### Added

-   Add `spectral_jax` argument to `UmapJax`, which defaults to `True` and uses jax for spectral layout when `init="spectral"`, the default for `UmapJax`.

## [0.0.2]

### Added

-   Add `move_other` argument to `optimize_layout_euclidean`.

### Changed

-   Optimization now updates the tail embedding in the attractive force calculation.

## [0.0.1]

### Added

-   Initial release
