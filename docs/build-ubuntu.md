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
./run configure  # or ./run c
./run build      # or ./run b
```
