# Building on Windows

## Prerequisites

- [Visual Studio 2022](https://visualstudio.microsoft.com/)

  Under **Workloads**, select the following item:

  - **Desktop & Mobile**

    - **Desktop development with C++**

  Under **Individual components**, select the following item:

  - **Code tools**

    - **Git for Windows**

- [Intel(R) oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)

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
./run init-vcpkg
./run cmake
./run build
```
