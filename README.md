# jizai

**jizai** is a fast and memory-efficient framework for radial basis function (RBF) interpolation.

## Features

- Interpolation of 1D, 2D, and 3D scattered data
- Surface reconstruction from 2.5D and 3D point clouds
- Fast kriging prediction ([dual kriging](https://github.com/unageek/jizai/wiki/Dual-kriging))
- Full control of absolute fitting tolerance and evaluation accuracy
- Fast and high-quality isosurface generation
  - Gradient search and surface tracking
  - Vertex position refinement
  - Vertex clustering
- Handling of 1M+ input points
- Inequality constraints
- Gradient constraints (Hermite–Birkhoff interpolation)

## Documentation

Please check out the [wiki](https://github.com/unageek/jizai/wiki).

## Platform Support

The following platforms are supported with the listed BLAS implementations:

| Platform | Architecture |    BLAS    |                                                                                        Build                                                                                         |
| :------: | :----------: | :--------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Windows  |     x64      |   oneMKL   |    [![Windows x64](https://github.com/unageek/jizai/actions/workflows/windows-x64.yml/badge.svg?branch=main)](https://github.com/unageek/jizai/actions/workflows/windows-x64.yml)    |
| Windows  |    ARM64     |   ArmPL    | [![Windows ARM64](https://github.com/unageek/jizai/actions/workflows/windows-arm64.yml/badge.svg?branch=main)](https://github.com/unageek/jizai/actions/workflows/windows-arm64.yml) |
|  macOS   |    ARM64     | Accelerate |    [![macOS ARM64](https://github.com/unageek/jizai/actions/workflows/macos-arm64.yml/badge.svg?branch=main)](https://github.com/unageek/jizai/actions/workflows/macos-arm64.yml)    |
|  Linux   |     x64      |   oneMKL   |       [![Linux x64](https://github.com/unageek/jizai/actions/workflows/linux-x64.yml/badge.svg?branch=main)](https://github.com/unageek/jizai/actions/workflows/linux-x64.yml)       |
|  Linux   |    ARM64     |   ArmPL    |    [![Linux ARM64](https://github.com/unageek/jizai/actions/workflows/linux-arm64.yml/badge.svg?branch=main)](https://github.com/unageek/jizai/actions/workflows/linux-arm64.yml)    |

oneMKL and ArmPL are automatically downloaded and extracted into the build tree during the configuration process.

## References

1. J. C. Carr, R. K. Beatson, J. B. Cherrie, T. J. Mitchell, W. R. Fright, B. C. McCallum, and T. R. Evans. Reconstruction and representation of 3D objects with radial basis functions. In _Computer Graphics SIGGRAPH 2001 proceedings_, ACM Press/ACM SIGGRAPH, pages 67–76, 12-17 August 2001. [https://doi.org/10.1145/383259.383266](https://doi.org/10.1145/383259.383266)

1. R. K. Beatson, W. A. Light, and S. Billings. Fast solution of the radial basis function interpolation equations: Domain decomposition methods. _SIAM J. Sci. Comput._, 22(5):1717–1740, 2000. [http://doi.org/10.1137/S1064827599361771](http://doi.org/10.1137/S1064827599361771)

1. G. M. Treece, R. W. Prager, and A. H. Gee. Regularised marching tetrahedra: improved iso-surface extraction. _Computers and Graphics_, 23(4):583–598, 1999. [https://doi.org/10.1016/S0097-8493(99)00076-X](<https://doi.org/10.1016/S0097-8493(99)00076-X>)
