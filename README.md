# Polatory

Polatory is a framework for fast 3D spline interpolation and kriging, developed by GSI Co. Ltd.

## License

Polatory is available under two different licenses:

* GNU General Public License v3.0 (GPLv3)
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

## Module index

| Module                 | Description                                      |
| ---------------------- | ------------------------------------------------ |
| common                 | Common utility functions and classes.            |
| distribution_generator | Random points generation for test cases.         |
| fmm                    | Fast multipole methods (wrapper of ScalFMM).     |
| geometry               | Geometric classes.                               |
| interpolation          | RBF fitting and evaluation.                      |
| isosurface             | Isosurface generation.                           |
| kriging                | Variogram estimation and validation.             |
| krylov                 | Krylov subspace methods.                         |
| numeric                | Robust algorithms.                               |
| point_cloud            | Scattered data generation from point clouds.     |
| polynomial             | Polynomial part of RBF interpolant.              |
| preconditioner         | The preconditioner used with Krylov subspace methods. |
| rbf                    | Definition of RBFs.                              |
| third_party            | Third party libraries.                           |

## Contribution

Contributions are welcome! You can contribute to this project in several ways:

### Star this Project

You can just click the ★Star button to show your interest.

### Open an Issue

Feel free to open an issue, if you have/found any questions, requests or issues.

### Send a Pull Request

You can fork and propose improvements in the source code. When sending a PR for the first time, please <a href="https://www.clahub.com/agreements/polatory/draft">review and sign the Contributor License Agreement</a>.

### Visit ScalFMM

The fast RBF evaluation exploits ScalFMM, a C++ library implements FMM algorithms, developed by Inria. If you find this project useful, you can also check the development of ScalFMM at https://gitlab.inria.fr/solverstack/ScalFMM/.
