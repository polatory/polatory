<div align="center">
  <img src="https://polatory.github.io/images/polatory_logo.png" width="400" alt="Polatory">
</div>

---

**Polatory** is a fast and memory-efficient framework for RBF (radial basis function) interpolation.

## Features

- Interpolation of 1D/2D/3D scattered data
- Surface reconstruction from 2.5D/3D point clouds
- Fast kriging prediction ([dual kriging](https://github.com/polatory/polatory/wiki/Dual-kriging))
- Quality isosurface generation
- Supports over 1M of input points
- Inequality constraints
- Gradient constraints (Hermite–Birkhoff interpolation)
- [List of basic functions](https://github.com/polatory/polatory/wiki/List-of-basic-functions)
- [Command-line interface](https://github.com/polatory/polatory/wiki/CLI-workflows)
- Python bindings (run `pip install .` after cloning)

## Building

[On Windows](docs/build-windows.md) • [On macOS](docs/build-macos.md) • [On Ubuntu](docs/build-ubuntu.md)

Polatory requires a C++ compiler that supports C++20 and OpenMP 5.0. Clang 17 is recommended on all platforms.

## Contribution

Contributions are welcome! You can contribute to this project in several ways:

### Star the Repo

Just click the <kbd>⭐️ Star</kbd> button on top of the page to show your interest!

### <a href="https://github.com/polatory/polatory/issues">File an Issue</a>

Do not hesitate to file an issue if you have any questions, feature requests, or if you have encountered unexpected results (please include a minimal reproducible example).

### <a href="https://github.com/polatory/polatory/pulls">Submit a Pull Request</a>

You can fork the repo and make some improvements, then feel free to submit a pull request!

## References

1. J. C. Carr, R. K. Beatson, J. B. Cherrie, T. J. Mitchell, W. R. Fright, B. C. McCallum, and T. R. Evans. Reconstruction and representation of 3D objects with radial basis functions. In _Computer Graphics SIGGRAPH 2001 proceedings_, ACM Press/ACM SIGGRAPH, pages 67–76, 12-17 August 2001. [https://doi.org/10.1145/383259.383266](https://doi.org/10.1145/383259.383266)

1. R. K. Beatson, W. A. Light, and S. Billings. Fast solution of the radial basis function interpolation equations: Domain decomposition methods. _SIAM J. Sci. Comput._, 22(5):1717–1740, 2000. [http://doi.org/10.1137/S1064827599361771](http://doi.org/10.1137/S1064827599361771)

1. G. M. Treece, R. W. Prager, and A. H. Gee. Regularised marching tetrahedra: improved iso-surface extraction. _Computers and Graphics_, 23(4):583–598, 1999. [https://doi.org/10.1016/S0097-8493(99)00076-X](<https://doi.org/10.1016/S0097-8493(99)00076-X>)
