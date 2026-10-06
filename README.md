# jizai

**jizai** (<ruby>自在<rp>(</rp><rt>じざい</rt><rp>)</rp></ruby>) is a fast and memory-efficient framework for radial basis function (RBF) interpolation.

## Features

- Interpolation of 1D, 2D, and 3D scattered data
- Fast geostatistical prediction
- Support for 1M+ input points
- Inequality and gradient constraints
- Surface reconstruction from 2.5D and 3D point clouds
- Advanced isosurface generation
  - Surface discovery and tracking
  - Vertex position refinement
  - Vertex clustering
  - Snapping to input points

## Installation

1. Install the [prerequisites](https://github.com/unageek/jizai/wiki/Prerequisites).

1. Install jizai from PyPI using pip:

   ```sh
   pip install jizai
   ```

   jizai is built from source during installation, which takes a while.

## Usage

Here is an example of surface reconstruction from a point cloud with normals.

```py
import urllib.request

import jizai as jz
import numpy as np

url = "https://www.cs.jhu.edu/~misha/Code/PoissonRecon/horse.npts"
with urllib.request.urlopen(url) as response:
    table = np.loadtxt(response)

indices = jz.DistanceFilter(table[:, :3]).filtered_indices()
points = table[indices, :3]
normals = table[indices, 3:]

gen = jz.SdfDataGenerator(points, normals)

rbf = jz.Biharmonic3D(dim=3)
model = jz.Model(rbf)
interp = jz.Interpolant(model)
interp.fit(gen.sdf_points, gen.sdf_values, tolerance=1e-5, accuracy=1e-7)

bbox = jz.Bbox(np.full(3, -0.1), np.full(3, 0.1))
field_fn = jz.RbfFieldFunction(interp, accuracy=1e-7)
mesh = jz.Isosurface(bbox, 5e-4).generate_from_seed_points(points, field_fn)
mesh.export_obj("horse.obj")
```

**NOTE:** The example downloads `horse.npts`, one of the sample data sets provided with [PoissonRecon](https://www.cs.jhu.edu/~misha/Code/PoissonRecon/). The data is not part of jizai and is not covered by its license. No license is stated for the data, so check the terms with its provider before using it for anything beyond trying out this example.

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

1. J. C. Carr, R. K. Beatson, J. B. Cherrie, T. J. Mitchell, W. R. Fright, B. C. McCallum, and T. R. Evans. Reconstruction and representation of 3D objects with radial basis functions. In _Proceedings of the 28th Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH '01)_, pages 67–76, 2001. [https://doi.org/10.1145/383259.383266](https://doi.org/10.1145/383259.383266)

1. R. K. Beatson, W. A. Light, and S. Billings. Fast solution of the radial basis function interpolation equations: Domain decomposition methods. _SIAM Journal on Scientific Computing_, 22(5):1717–1740, 2001. [https://doi.org/10.1137/S1064827599361771](https://doi.org/10.1137/S1064827599361771)

1. G. M. Treece, R. W. Prager, and A. H. Gee. Regularised marching tetrahedra: improved iso-surface extraction. _Computers & Graphics_, 23(4):583–598, 1999. [https://doi.org/10.1016/S0097-8493(99)00076-X](<https://doi.org/10.1016/S0097-8493(99)00076-X>)
