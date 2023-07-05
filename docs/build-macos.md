# Building on macOS

## Prerequisites

- Xcode Command Line Tools

  ```bash
  xcode-select --install
  ```

- [Homebrew](https://brew.sh)

- Build tools

  ```bash
  brew install cmake llvm ninja
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
./run init-vcpkg
./run cmake
./run build
```
