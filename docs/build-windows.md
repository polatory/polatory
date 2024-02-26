# Building on Windows

## Prerequisites

- [Visual Studio 2022](https://visualstudio.microsoft.com/) 17.9 or later

  Under **Workloads**, select the following item:

  - **Desktop & Mobile**

    - **Desktop development with C++**

  Under **Individual components**, select the following item:

  - **Code tools**

    - **Git for Windows**

  - **Compilers, build tools, and runtimes**

    - **C++ Clang Compiler for Windows**

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

**NOTE:** To distribute the example programs, you need to install the [MSVC runtime library](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) and copy the OpenMP runtime library located at `<Visual Studio 2022 Installation Dir>\VC\Redist\MSVC\<version>\debug_nonredist\x64\Microsoft.VC143.OpenMP.LLVM\libomp140.x86_64.dll` to the same directory as the executables.
