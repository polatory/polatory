# Building on Ubuntu 22.04

1. Install build tools

   ```bash
   sudo apt install build-essential cmake curl git ninja-build pkg-config unzip
   ```

   If you use Clang, `libomp-dev` is required.

   ```bash
   sudo apt install clang libomp-dev
   ```

1. Install [Intel(R) oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html).

   ```bash
   wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \ | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
   echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
   sudo apt update
   sudo apt install intel-oneapi-mkl-devel
   ```

1. Clone Polatory

   ```bash
   git clone --recurse-submodules https://github.com/polatory/polatory.git
   cd polatory
   ```

   To update an existing repository:

   ```bash
   git pull --recurse-submodules
   ```

1. Build polatory

   ```bash
   ./run init-vcpkg
   ./run cmake
   ./run build
   ```
