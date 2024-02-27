# Building on Ubuntu 22.04

## Prerequisites

- Build tools

  ```bash
  sudo apt install build-essential cmake curl git ninja-build pkg-config unzip
  ```

- Clang 17

  ```bash
  wget https://apt.llvm.org/llvm.sh
  chmod +x llvm.sh
  sudo ./llvm.sh 17 all
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
