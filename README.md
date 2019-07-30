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
| Ubuntu 16.04 LTS | GCC 5.4 and Clang 3.8 |
| Ubuntu 18.04 LTS | GCC 7.3 and Clang 6.0 |
| Windows          | Visual Studio 2017    |

### Kriging via RBF Interpolation ([Benchmark](https://github.com/polatory/polatory/wiki/Benchmark))

Polatory can perform global kriging prediction via RBF interpolation. Although different terminology is used, both methods produce the same results. Here is the correspondence between kriging and RBF interpolation:

| Kriging             | RBF interpolation                          |
| ------------------- | ------------------------------------------ |
| Prediction          | Interpolation (fitting + evaluation)       |
| Covariance function | RBF                                        |
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

    On Ubuntu 16.04 LTS, CMake >= 3.9 must be installed manually.

    ```bash
    sudo apt install build-essential cmake git ninja-build
    ```
    If you use Clang, `libomp-dev` is required.
    ```bash
    sudo apt install clang libomp-dev
    ```

1. Download and install [Intel(R) MKL](https://software.intel.com/mkl).

    See https://software.intel.com/articles/installing-intel-free-libs-and-python-apt-repo for details.

    ```bash
    cd
    wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB -O - | sudo apt-key add -
    sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
    sudo apt update
    sudo apt install intel-mkl-64bit-2018.3-051
    ```

1. Install [Eigen](http://eigen.tuxfamily.org/)

    ```bash
    sudo apt install libeigen3-dev
    ```

1. Install [Google Test](https://github.com/google/googletest)

    ```bash
    git clone https://github.com/google/googletest.git
    cd googletest
    mkdir build && cd build
    cmake .. -GNinja
    ninja
    sudo ninja install
    ```

1. Install [Ceres Solver](http://ceres-solver.org/)

    ```bash
    sudo apt install libgoogle-glog-dev
    cd
    git clone https://ceres-solver.googlesource.com/ceres-solver
    cd ceres-solver
    mkdir build && cd build/
    cmake .. -GNinja -DCMAKE_LIBRARY_PATH=/opt/intel/mkl/lib/intel64 -DGFLAGS=OFF -DLAPACK=ON
    ninja
    sudo ninja install
    ```

1. Install [FLANN](http://www.cs.ubc.ca/research/flann/)

    ```bash
    sudo apt install libflann-dev
    ```

1. Download and build [Boost](http://www.boost.org/)

    ```bash
    cd
    wget https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.bz2
    tar xjf boost_1_69_0.tar.bz2
    cd boost_1_69_0
    ./bootstrap.sh
    ./b2 install -j8 --prefix=.
    ```

1. Build polatory

    ```bash
    cd
    git clone https://github.com/polatory/polatory.git
    cd polatory
    mkdir build && cd build
    cmake .. -GNinja -DBOOST_ROOT=~/boost_1_69_0
    ninja
    ```

### On Windows

1. Install Visual Studio Community 2017

    https://www.visualstudio.com/

    From the **Workloads** tab, select the following item.

    - Desktop development with C++

    From the **Individual components** tab, select the following item.

    - Code tools > Git for Windows

1. Install libraries with [vcpkg](https://github.com/Microsoft/vcpkg)

    ```bat
    cd /d C:\
    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    bootstrap-vcpkg.bat
    vcpkg install boost ceres flann eigen3 gtest --triplet x64-windows
    ```

    To update vcpkg and installed libraries, run the following commands:

    ```bat
    cd /d C:\vcpkg
    git pull
    bootstrap-vcpkg.bat
    vcpkg update
    vcpkg upgrade
    vcpkg upgrade --no-dry-run
    ```

1. [Download and install Intel(R) MKL](https://software.intel.com/mkl)

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
| third_party            | Third party libraries.                           |

## Acknowledgements

Polatory utilizes the following libraries:

| Library                                                | License                    | Used                  |
| ------------------------------------------------------ | -------------------------- | --------------------- |
| [Boost](http://www.boost.org/)                         | Boost Software License 1.0 | In several modules    |
| [Ceres Solver](http://ceres-solver.org/)               | The 3-clause BSD license   | In kriging module     |
| [Eigen](http://eigen.tuxfamily.org/)                   | Mozilla Public License 2.0 | In almost all modules |
| [FLANN](http://www.cs.ubc.ca/research/flann/)          | The 2-clause BSD license   | In point_cloud module |
| [Google Test](https://github.com/google/googletest)    | The 3-clause BSD license   | For unit testing      |
| [Intel(R) MKL](https://software.intel.com/mkl)         | <a href="https://software.intel.com/license/intel-simplified-software-license">Intel Simplified Software License</a> | As a backend for Ceres Solver, Eigen and ScalFMM |
| [ScalFMM](https://gitlab.inria.fr/solverstack/ScalFMM) | <a href="http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html">The CeCILL-C license</a> | In fmm module |
