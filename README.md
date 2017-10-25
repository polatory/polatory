<div align="center">
  <img src="https://polatory.github.io/images/polatory_logo.png" width="400" alt="Polatory">
</div>

----

Polatory is a fast and memory-efficient framework for RBF (radial basis function) interpolation, developed by [GSI Co. Ltd.](http://gsinet.co.jp/)

NOTE: This is a pre-release version. APIs subject to change without notice.

## Features

* Fast spline surface reconstruction of 2.5D/3D point cloud
* Fast interpolation (global kriging prediction) of 1D/2D/3D scattered data
* Meshing isosurfaces
* Supports large number (millions) of input points
* Supports globally supported RBFs

### [Benchmark](https://github.com/polatory/polatory/wiki/Benchmark)

### Kriging via RBF interpolation

Polatory can perform global kriging prediction via RBF interpolation. Here is the correspondence between kriging prediction and RBF interpolation:

| Kriging prediction  | RBF interpolation                          |
| ------------------- | ------------------------------------------ |
| Prediction          | Interpolation (fitting + evaluation)       |
| Covariance function | RBF                                        |
| Nugget effect       | Spline smoothing                           |
| Simple kriging      | Interpolant with no polynomial             |
| Ordinary kriging    | Interpolant with polynomial of degree 0    |
| Universal kriging   | Interpolant with polynomial of degree >= 1 |
| Weights             | (Not computed)                             |
| Standard errors     | (Not computed)                             |

**Rapidly decaying covariance functions are not supported in the current release, such as the spherical model and the Gaussian model.** These RBFs are included for reference purposes. The spherical model can be substituted by `cov_quasi_spherical9`, which is quite similar to the original one.

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
   sudo apt-get install intel-mkl-64bit-2017.4-061
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
   sudo apt-get install libgoogle-glog-dev
   cd
   git clone https://ceres-solver.googlesource.com/ceres-solver
   cd ceres-solver
   mkdir build; cd build/
   cmake .. -DCMAKE_LIBRARY_PATH=/opt/intel/mkl/lib/intel64 -DGFLAGS=OFF -DLAPACK=ON
   make -j8
   sudo make install
   ```

1. Install [FLANN](http://www.cs.ubc.ca/research/flann/)
   ```bash
   sudo apt-get install libflann-dev
   ```

1. Download and extract [Boost](http://www.boost.org/)

   Polatory currently uses header-only libraries of boost, so you don't have to build it.
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

### On Windows

1. Install libraries
   ```
   cd /d C:
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   ./vcpkg install boost:x64-windows ceres:x64-windows flann:x64-windows eigen3:x64-windows gtest:x64-windows
   ```

1. Build polatory
   ```
   cd /d %userprofile%
   git clone https://github.com/polatory/polatory.git
   cd polatory
   mkdir build
   cd build
   cmake .. -G"Visual Studio 15 2017 Win64" -T"Intel C++ Compiler 17.0" -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_ROOT=C:/vcpkg/installed/x64-windows
   msbuild polatory.sln /p:Configuration=Release
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
| io                     | Reader and writer for table data.                |
| isosurface             | Isosurface generation.                           |
| kriging                | Parameter estimation and validation for kriging. |
| krylov                 | Krylov subspace methods.                         |
| numeric                | Numerical utilities.                             |
| point_cloud            | Scattered data generation from point clouds.     |
| polynomial             | Polynomial part of RBF interpolant.              |
| preconditioner         | The preconditioner used with Krylov subspace methods. |
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
