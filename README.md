<div align="center">
  <img src="https://polatory.github.io/images/polatory_logo.png" width="400" alt="Polatory">
</div>

---

**Polatory** is a fast and memory-efficient framework for RBF (radial basis function) interpolation.

NOTE: This is a pre-release version. APIs subject to change without notice.

[Features](#features) • [Building](#building) • [Contribution](#contribution) • [Module Index](#module-index) • [Acknowledgements](#acknowledgements)

## Features

- Fast spline surface reconstruction from 2.5D/3D point cloud
- Fast interpolation of 1D/2D/3D scattered data (kriging prediction)
- Meshing isosurfaces
- Supports large number (millions) of input points
- Supports inequality constraints
- [List of available RBFs](https://github.com/polatory/polatory/wiki/List-of-Available-RBFs)

### Platforms

Polatory runs on x86-64 processors and continuously tested on the following platforms.

| OS               | Toolchain             |
| ---------------- | --------------------- |
| Ubuntu 18.04 LTS | GCC 7.4 and Clang 6.0 |
| Windows          | Visual Studio 2019    |

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
| Universal kriging                            | Interpolant with polynomial of degree >= 1            |
| Weights                                      | (Not computed) Cardinal basis functions               |
| Kriging variance                             | (Not computed)                                        |

A limited number of covariance functions are supported. See the [list of available RBFs](https://github.com/polatory/polatory/wiki/List-of-Available-RBFs) for details.

## Building

[On Ubuntu](docs/build-ubuntu.md) • [On Windows](docs/build-windows.md)

## Contribution

Your contribution is appreciated! You can contribute to this project in several ways:

### Star the Repo

You can just click the [★ Star] button on top of the page to show your interest.

### <a href="https://github.com/polatory/polatory/issues">File an Issue</a>

Feel free to file an issue, if you have any questions, feature requests, or if you have found any unexpected results (please include a minimal reproducible example).

### <a href="https://github.com/polatory/polatory/pulls">Create a Pull Request</a>

You can fork the source tree and make some improvements to it. Then feel free to create a PR.

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
