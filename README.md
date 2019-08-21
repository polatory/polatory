<div align="center">
  <img src="https://polatory.github.io/images/polatory_logo.png" width="400" alt="Polatory">
</div>

----

[![Travis CI](https://travis-ci.org/polatory/polatory.svg?branch=master)](https://travis-ci.org/polatory/polatory)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/andjvgjr58axrbe0/branch/master?svg=true)](https://ci.appveyor.com/project/mizuno-gsinet/polatory/branch/master)

Polatory is a fast and memory-efficient framework for RBF (radial basis function) interpolation, developed by [GSI Co. Ltd.](http://gsinet.co.jp/)

NOTE: This is a pre-release version. APIs subject to change without notice.

[Features](#features) | [License](#license) | [Building](#building) | [Contribution](#contribution) | [Module Index](#module-index) | [Acknowledgements](#acknowledgements)

## Features

* Fast spline surface reconstruction from 2.5D/3D point cloud
* Fast interpolation of 1D/2D/3D scattered data (kriging prediction)
* Meshing isosurfaces
* Supports large number (millions) of input points
* Supports inequality constraints
* [List of available RBFs](docs/rbf.ipynb)

### Platforms

Polatory runs on x86-64 processors and continuously tested on the following platforms.

| OS               | Toolchain             |
| ---------------- | --------------------- |
| Ubuntu 18.04 LTS | GCC 7.4 and Clang 6.0 |
| Windows          | Visual Studio 2017    |

### Kriging via RBF Interpolation ([Benchmark](https://github.com/polatory/polatory/wiki/Benchmark))

Polatory can perform global kriging prediction via RBF interpolation. Although different terminology is used, both methods produce the same results. Here is the correspondence between kriging and RBF interpolation:

| Kriging             | RBF interpolation                          |
| ------------------- | ------------------------------------------ |
| Prediction          | Interpolation (fitting + evaluation)       |
| Covariance function | Positive definite RBF                      |
| Nugget effect       | Spline smoothing                           |
| Simple kriging      | Interpolant with no polynomial             |
| Ordinary kriging    | Interpolant with polynomial of degree 0    |
| Universal kriging   | Interpolant with polynomial of degree >= 1 |
| Weights             | (Not computed) Cardinal basis functions    |
| Kriging variance    | (Not computed)                             |

Rapidly decaying covariance functions are not supported in the current version, such as the spherical and the Gaussian models. The spherical model can be substituted by a similar one named `cov_quasi_spherical9`.

## License

Polatory is available under two different licenses:

* <a href="https://opensource.org/licenses/gpl-3.0.html">GNU General Public License version 3</a> for **non-commercial** use
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
    sudo apt install intel-mkl-64bit-2018.3-051
    ```

1. Install libraries with [vcpkg](https://github.com/Microsoft/vcpkg)

    ```bash
    cd
    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    ./bootstrap-vcpkg.sh
    ./vcpkg install abseil boost-filesystem boost-program-options boost-serialization ceres eigen3 flann gsl-lite gtest --triplet x64-linux
    ```

    See also: [Updating vcpkg](https://github.com/polatory/polatory/wiki/Updating-vcpkg)

1. Build polatory

    ```bash
    cd
    git clone https://github.com/polatory/polatory.git
    cd polatory
    mkdir build && cd build
    cmake .. -GNinja -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake
    ninja
    ```

### On Windows

1. Install Visual Studio Community 2017

    https://www.visualstudio.com/

    From the **Workloads** tab, select the following item.

    - Desktop development with C++

    From the **Individual components** tab, select the following item.

    - Code tools > Git for Windows

1. [Download and install Intel(R) MKL](https://software.intel.com/mkl)

1. Install libraries with [vcpkg](https://github.com/Microsoft/vcpkg)

    ```bat
    cd /d C:\
    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    bootstrap-vcpkg.bat
    vcpkg install abseil boost-filesystem boost-program-options boost-serialization ceres eigen3 flann gsl-lite gtest --triplet x64-windows
    ```

    See also: [Updating vcpkg](https://github.com/polatory/polatory/wiki/Updating-vcpkg)

1. Build polatory

    Open **Start** > **Visual Studio 2017** > **x64 Native Tools Command Prompt for VS 2017**.

    ```bat
    cd /d %userprofile%
    git clone https://github.com/polatory/polatory.git
    cd polatory
    mkdir build
    cd build
    cmake .. -GNinja -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
    ninja
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
