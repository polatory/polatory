<div align="center">
  <img src="https://polatory.github.io/images/polatory_logo.png" width="400" alt="Polatory">
</div>

---

**Polatory** is a fast and memory-efficient framework for RBF (radial basis function) interpolation.

NOTE: This is a pre-release version. APIs are subject to change without notice.

[Features](#features) • [Building](#building) • [Contribution](#contribution) • [Module Index](#module-index) • [References](#references)

## Features

- Fast spline surface reconstruction from 2.5D/3D point cloud
- Fast interpolation of 1D/2D/3D scattered data (kriging prediction)
- Meshing isosurfaces
- Supports large number (millions) of input points
- Supports inequality constraints
- [List of available RBFs](https://github.com/polatory/polatory/wiki/List-of-Available-RBFs)

### Supported Compilers

Polatory requires a C++ compiler that supports C++20 and OpenMP 2.0.

### Kriging via RBF Interpolation ([Benchmark](https://github.com/polatory/polatory/wiki/Benchmark))

Polatory can perform kriging prediction via RBF interpolation (dual kriging). Although different terminologies are used, both methods produce the same results. Here is the correspondence between them:

| Kriging                                      | RBF interpolation                                     |
| -------------------------------------------- | ----------------------------------------------------- |
| Prediction                                   | Interpolation (fitting + evaluation)                  |
| Covariance function                          | RBF, positive definite                                |
| Generalized covariance function of order _k_ | RBF, conditionally positive definite of order _k_ + 1 |
| Nugget effect model                          | Spline smoothing                                      |
| Simple kriging                               | Interpolant with no polynomial                        |
| Ordinary kriging                             | Interpolant with polynomial of degree 0               |
| Universal kriging                            | Interpolant with polynomial of degree ≥ 1             |
| Weights                                      | (Not computed) Weights of cardinal basis functions    |
| Kriging variance                             | (Not computed)                                        |

A limited number of covariance functions are supported. See the [list of available RBFs](https://github.com/polatory/polatory/wiki/List-of-Available-RBFs) for details.

## Building

[On Ubuntu](docs/build-ubuntu.md) • [On Windows](docs/build-windows.md)

## Contribution

Contributions are welcome! You can contribute to this project in several ways:

### Star the Repo

Just click the [★ Star] button on top of the page to show your interest!

### <a href="https://github.com/polatory/polatory/issues">File an Issue</a>

Do not hesitate to file an issue if you have any questions, feature requests, or if you have encountered unexpected results (please include a minimal reproducible example).

### <a href="https://github.com/polatory/polatory/pulls">Submit a Pull Request</a>

You can fork the repo and make some improvements, then feel free to submit a pull request!

## Module Index

| Module         | Description                                           |
| -------------- | ----------------------------------------------------- |
| common         | Common utility functions and classes.                 |
| fmm            | Fast multipole methods (wrapper of ScalFMM).          |
| geometry       | Geometric utilities.                                  |
| interpolation  | RBF fitting and evaluation.                           |
| isosurface     | Isosurface generation.                                |
| kriging        | Parameter estimation and validation for kriging.      |
| krylov         | Krylov subspace methods.                              |
| numeric        | Numerical utilities.                                  |
| point_cloud    | SDF data generation from point clouds.                |
| polynomial     | Polynomial part of RBF interpolant.                   |
| preconditioner | The preconditioner used with Krylov subspace methods. |
| rbf            | Definition of RBFs/covariance functions.              |

## References

1. J. C. Carr, R. K. Beatson, J. B. Cherrie, T. J. Mitchell, W. R. Fright, B. C. McCallum, and T. R. Evans. Reconstruction and representation of 3D objects with radial basis functions. In _Computer Graphics SIGGRAPH 2001 proceedings_, ACM Press/ACM SIGGRAPH, pages 67–76, 12-17 August 2001. [https://doi.org/10.1145/383259.383266](https://doi.org/10.1145/383259.383266)

1. R. K. Beatson, W. A. Light, and S. Billings. Fast solution of the radial basis function interpolation equations: Domain decomposition methods. _SIAM J. Sci. Comput._, 22(5):1717–1740, 2000. [http://doi.org/10.1137/S1064827599361771](http://doi.org/10.1137/S1064827599361771)

1. G. M. Treece, R. W. Prager, and A. H. Gee. Regularised marching tetrahedra: improved iso-surface extraction. _Computers and Graphics_, 23(4):583–598, 1999. [https://doi.org/10.1016/S0097-8493(99)00076-X](<https://doi.org/10.1016/S0097-8493(99)00076-X>)
