<div align="center">
  <img src="https://polatory.github.io/images/polatory_logo.png" width="400" alt="Polatory">
</div>

----

[![Travis CI](https://travis-ci.org/polatory/polatory.svg?branch=master)](https://travis-ci.org/polatory/polatory)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/andjvgjr58axrbe0/branch/master?svg=true)](https://ci.appveyor.com/project/mizuno-gsinet/polatory/branch/master)
[![Read the Docs](https://readthedocs.org/projects/polatory/badge/?version=latest)](https://polatory.readthedocs.io/en/latest/?badge=latest)

Polatory is a fast and memory-efficient framework for RBF (radial basis function) interpolation, developed by [GSI Co., Ltd.](http://gsinet.co.jp/)

NOTE: This is a pre-release version. APIs subject to change without notice.

[Features](#features) | [License](#license) | [Building](#building) | [Contribution](#contribution) | [Module Index](#module-index) | [Acknowledgements](#acknowledgements)

## Features

* Fast spline surface reconstruction from 2.5D/3D point cloud
* Fast interpolation of 1D/2D/3D scattered data (kriging prediction)
* Meshing isosurfaces
* Supports large number (millions) of input points
* Supports inequality constraints
* [List of available RBFs](https://polatory.readthedocs.io/en/latest/rbfs.html)

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
| Generalized covariance function of order *k* | RBF, conditionally positive definite of order *k* + 1 |
| Nugget effect model                          | Spline smoothing                                      |
| Simple kriging                               | Interpolant with no polynomial                        |
| Ordinary kriging                             | Interpolant with polynomial of degree 0               |
| Universal kriging                            | Interpolant with polynomial of degree >= 1            |
| Weights                                      | (Not computed) Cardinal basis functions               |
| Kriging variance                             | (Not computed)                                        |

A limited number of covariance functions are supported. See the [list of available RBFs](https://polatory.readthedocs.io/en/latest/rbfs.html) for details.

## License

Polatory is available under two different licenses:

* [GNU General Public License, version 3](LICENSE.GPLv3)
* Commercial license (please contact at mizuno(at)gsinet.co.jp)

## Building

### On Ubuntu

1. Install build tools

    ```bash
    sudo apt install build-essential cmake curl git ninja-build unzip
    ```

    If you use Clang, `libomp-dev` is required.

    ```bash
    sudo apt install clang libomp-dev
    ```

1. Download and install [Intel(R) MKL](https://software.intel.com/mkl).

    See https://software.intel.com/articles/installing-intel-free-libs-and-python-apt-repo for details.

    ```bash
    wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB -O - | sudo apt-key add -
    sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
    sudo apt update
    sudo apt install intel-mkl-64bit-2019.5-075
    ```

1. Clone Polatory

    ```bash
    git clone --recursive https://github.com/polatory/polatory.git
    cd polatory
    ```

    To update an existing repository:

    ```bash
    git submodule sync
    git submodule update --init --recursive
    ```

1. Build polatory

    ```bash
    ./run init-vcpkg
    ./run cmake
    ./run build
    ```

### On Windows

1. Install Visual Studio Community 2019

    https://www.visualstudio.com/

    Under the **Workloads** tab, select the following item.

    - Desktop development with C++

    Under the **Individual components** tab, select the following item.

    - Code tools > Git for Windows

1. [Download and install Intel(R) MKL](https://software.intel.com/mkl)

1. Clone Polatory

    ```pwsh
    git clone --recursive https://github.com/polatory/polatory.git
    cd polatory
    ```

    To update an existing repository:

    ```pwsh
    git submodule sync
    git submodule update --init --recursive
    ```

1. Build polatory

    ```pwsh
    .\run init-vcpkg
    .\run cmake
    .\run build
    ```

## Contribution

We welcome your contributions! You can contribute to this project in several ways:

### Add a Star

You can just click the ★Star button to show your interest.

### <a href="https://github.com/polatory/polatory/issues">File an Issue</a>

Feel free to file an issue, if you have any questions, feature requests, or if you have found any unexpected results (please include a minimal reproducible example).

### <a href="https://github.com/polatory/polatory/pulls">Create a Pull Request</a>

You can fork the source tree and make some improvements to it. Then feel free to create a PR. When sending a PR for the first time, please <a href="https://cla-assistant.io/polatory/polatory">review and sign the Individual Contributor License Agreement</a>.

## Module Index

| Module                 | Description                                      |
| ---------------------- | ------------------------------------------------ |
| common                 | Common utility functions and classes.            |
| fmm                    | Fast multipole methods (wrapper of ScalFMM).     |
| geometry               | Geometric utilities.                             |
| interpolation          | RBF fitting and evaluation.                      |
| isosurface             | Isosurface generation.                           |
| kriging                | Parameter estimation and validation for kriging. |
| krylov                 | Krylov subspace methods.                         |
| numeric                | Numerical utilities.                             |
| point_cloud            | SDF data generation from point clouds.           |
| polynomial             | Polynomial part of RBF interpolant.              |
| preconditioner         | The preconditioner used with Krylov subspace methods. |
| rbf                    | Definition of RBFs/covariance functions.         |

## Acknowledgements

Polatory is built upon the following libraries. Each library may have other dependencies.

<dl>
  <dt><a href="https://abseil.io/">Abseil</a></dt>
  <dd>Apache License 2.0</dd>
  <dt><a href="http://www.boost.org/">Boost</a></dt>
  <dd>Boost Software License 1.0</dd>
  <dt><a href="http://ceres-solver.org/">Ceres Solver</a></dt>
  <dd>BSD 3-Clause License</dd>
  <dt><a href="https://github.com/google/double-conversion">double-conversion</a></dt>
  <dd>BSD 3-Clause License</dd>
  <dt><a href="http://eigen.tuxfamily.org/">Eigen</a></dt>
  <dd>Mozilla Public License 2.0</dd>
  <dt><a href="http://www.cs.ubc.ca/research/flann/">FLANN</a></dt>
  <dd>BSD 2-Clause License</dd>
  <dt><a href="https://github.com/google/googletest">Google Test</a></dt>
  <dd>BSD 3-Clause License</dd>
  <dt><a href="https://github.com/martinmoene/gsl-lite">gsl-lite</a></dt>
  <dd>MIT License</dd>
  <dt><a href="https://software.intel.com/mkl">Intel(R) MKL</a></dt>
  <dd><a href="https://software.intel.com/license/intel-simplified-software-license">Intel Simplified Software License</a></dd>
  <dt><a href="https://gitlab.inria.fr/solverstack/ScalFMM">ScalFMM</a></dt>
  <dd><a href="http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html">CeCILL-C License 1.0</a></dd>
</dl>
