# Building on Ubuntu 22.04

## Prerequisites

- Build tools

  ```bash
  sudo apt install build-essential cmake curl git ninja-build pkg-config unzip
  ```

  If you prefer Clang:

  ```bash
  sudo apt install clang libomp-dev
  ```

- [Intel(R) oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)

  ```bash
  wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \ | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
  echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
  sudo apt update
  sudo apt install intel-oneapi-mkl-devel
  ```

  See [this page](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2023-1/apt.html) for more details.

## Clone

```bash
git clone --recurse-submodules https://github.com/polatory/polatory.git
cd polatory
```

To update an existing repository:

```bash
git pull --recurse-submodules
```

## Build

```bash
./run init-vcpkg
./run cmake
./run build
```
