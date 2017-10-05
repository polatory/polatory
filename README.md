# Polatory

Polatory is a fast and memory-efficient framework for spline surface reconstruction and spatial interpolation, developed by [GSI Co. Ltd.](http://gsinet.co.jp/)

## What Can It Do

* Fast spline surface reconstruction of 2.5D/3D point cloud
* Fast interpolation (kriging prediction) of 3D scattered data
* Meshing isosurfaces
* Supports large number (~1M) of input points
* Supports globally supported RBFs
* Supports user-defined smooth, non-oscillatory RBFs.

### Correspondence between kriging and RBF interpolation

| Kriging             | RBF interpolation                          |
| ------------------- | ------------------------------------------ |
| Covariance function | RBF                                        |
| Nugget effect       | Spline smoothing                           |
| Simple kriging      | Interpolant with no polynomial             |
| Ordinary kriging    | Interpolant with polynomial of degree 0    |
| Universal kriging   | Interpolant with polynomial of degree >= 1 |

## License

Polatory is available under two different licenses:

* GNU General Public License v3.0 (GPLv3) for non-commercial use
* Commercial license (please contact at mizuno(at)gsinet.co.jp)

## Platforms

Polatory is available on following platforms/toolchains.

### Linux

Ubuntu 16.04 LTS / GCC 5.4 and Clang 3.8

### Windows

Visual Studio 2017 / Intel Parallel Studio XE 2017

## Building

### On Ubuntu

1. Install build tools
   ```bash
   sudo apt-get install build-essential cmake git
   ```
   If you use Clang, Intel OpenMP needs to be installed.
   ```bash
   sudo apt-get install clang-3.8 libiomp-dev
   ```

1. Download and install Intel MKL.

   See https://software.intel.com/articles/installing-intel-free-libs-and-python-apt-repo for details.
   ```bash
   cd
   wget http://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
   sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
   sudo sh -c 'echo deb http://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
   sudo apt-get update
   sudo apt-get install intel-mkl-64bit-2017.3-056
   ```

1. Install [Eigen](http://eigen.tuxfamily.org/)
   ```bash
   sudo apt-get install libeigen3-dev
   ```

1. Install [Google Test](https://github.com/google/googletest)
   ```bash
   sudo apt-get install libgtest-dev
   cd
   mkdir gtest-build; cd gtest-build/
   cmake /usr/src/gtest/
   make
   sudo cp *.a /usr/local/lib/
   ```

1. Install [Ceres Solver](http://ceres-solver.org/)
   ```bash
   cd
   git clone https://ceres-solver.googlesource.com/ceres-solver
   cd ceres-solver
   mkdir build; cd build/
   cmake .. -DCMAKE_LIBRARY_PATH=/opt/intel/mkl/lib/intel64 -DGFLAGS=OFF -DLAPACK=ON -DMINIGLOG=ON
   make -j8
   sudo make install
   ```

1. Download and extract [Boost](http://www.boost.org/)

   Polatory currently uses header-only libraries from boost, so you don't have to build it.
   ```bash
   cd
   wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.bz2
   tar xvfj boost_1_64_0.tar.bz2
   ```

1. Build polatory
   ```bash
   cd
   git clone https://github.com/polatory/polatory.git
   cd polatory
   mkdir build; cd build
   cmake .. -DBOOST_ROOT=~/boost_1_64_0 -DCMAKE_BUILD_TYPE=Release
   make -j8
   ```

## Contribution

We welcome your contributions! You can contribute to this project in several ways:

### Star this Project

You can just click the ★Star button to show your interest.

### <a href="https://github.com/polatory/polatory/issues">Create an Issue</a>

Feel free to create an issue, if you have any questions, requests, or if you have found any issues (please include a minimal reproducible example).

### <a href="https://github.com/polatory/polatory/pulls">Create a Pull Request</a>

You can fork the source tree and make some improvements to it. Then feel free to create a PR. When sending a PR for the first time, please <a href="https://cla-assistant.io/polatory/polatory">review and sign the Individual Contributor License Agreement</a>.

## Module index

| Module                 | Description                                      |
| ---------------------- | ------------------------------------------------ |
| common                 | Common utility functions and classes.            |
| fmm                    | Fast multipole methods (wrapper of ScalFMM).     |
| geometry               | Geometric utilities.                             |
| interpolation          | RBF fitting and evaluation.                      |
| isosurface             | Isosurface generation.                           |
| kriging                | Parameter estimation and validation for kriging. |
| krylov                 | Krylov subspace methods.                         |
| numeric                | Numerically robust algorithms.                   |
| point_cloud            | Scattered data generation from point clouds.     |
| polynomial             | Polynomial part of RBF interpolant.              |
| preconditioner         | The preconditioner used with Krylov subspace methods. |
| random_points          | Random points generation for unit testing.       |
| rbf                    | Definition of RBFs/covariance functions.         |
| third_party            | Third party libraries.                           |

## Acknowledgements

Polatory utilizes the following libraries:

* [Boost](http://www.boost.org/)
  
  <dl>
    <dt>License</dt>
    <dd>Boost Software License</dd>
    <dt>Used in</dt>
    <dd>Several modules</dd>
  </dl>

* [Ceres Solver](http://ceres-solver.org/)
  
  <dl>
    <dt>License</dt>
    <dd>The 3-clause BSD license</dd>
    <dt>Used in</dt>
    <dd>kriging module</dd>
  </dl>

* [Eigen](http://eigen.tuxfamily.org/)
  
  <dl>
    <dt>License</dt>
    <dd>MPL2</dd>
    <dt>Used in</dt>
    <dd>Almost everywhere</dd>
  </dl>

* [FLANN](http://www.cs.ubc.ca/research/flann/)
  
  <dl>
    <dt>License</dt>
    <dd>The 2-clause BSD license</dd>
    <dt>Used in</dt>
    <dd>point_cloud module</dd>
  </dl>

* [Google Test](https://github.com/google/googletest)
  
  <dl>
    <dt>License</dt>
    <dd>The 3-clause BSD license</dd>
    <dt>Used in</dt>
    <dd>Unit testing</dd>
  </dl>

* [ScalFMM](https://gitlab.inria.fr/solverstack/ScalFMM)
  
  <dl>
    <dt>License</dt>
    <dd>The CeCILL-C license</dd>
    <dt>Used in</dt>
    <dd>fmm module</dd>
  </dl>
