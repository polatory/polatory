# Building on Windows

## Prerequisites

- [Visual Studio 2022](https://visualstudio.microsoft.com/)

  Under **Workloads**, select the following item:

  - **Desktop & Mobile**

    - **Desktop development with C++**

  Under **Individual components**, select the following item:

  - **Code tools**

    - **Git for Windows**

- Clang 17 (must be added to PATH)

## Clone

```pwsh
git clone --recurse-submodules https://github.com/polatory/polatory.git
cd polatory
```

To update an existing repository:

```pwsh
git pull --recurse-submodules
```

## Build

```pwsh
./run configure  # or ./run c
./run build      # or ./run b
```
