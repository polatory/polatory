List of Available RBFs
======================

In the following definitions, all scaling parameters are fixed to 1.

Splines
-------

biharmonic2d
^^^^^^^^^^^^

The basis function for biharmonic spline interpolation of 2D data. It can also be used for generic interpolation of any dimensional data.

.. math::
   \phi(r) = \begin{cases}
     0 & r = 0 \\
     r^2 \ln r & \mathrm{otherwise}
    \end{cases}

:math:`\phi(\|\cdot\|)` is conditionally positive definite of order 2 on every :math:`\mathbb{R}^d`. Thus the interpolant requires a polynomial part of degree :math:`\geq 1`.

biharmonic3d
^^^^^^^^^^^^

The basis function for biharmonic spline interpolation of 3D data. It can also be used for generic interpolation of any dimensional data.

.. math::
   \phi(r) = -r

:math:`\phi(\|\cdot\|)` is conditionally positive definite of order 1 on every :math:`\mathbb{R}^d`. Thus the interpolant requires a polynomial of degree :math:`\geq 0`.


Covariance functions
--------------------

cov_exponential
^^^^^^^^^^^^^^^

The exponential model.

.. math::
   C(r) = \exp(-r)

:math:`C(\|\cdot\|)` is positive definite on every :math:`\mathbb{R}^d`.

cov_spheroidal3
^^^^^^^^^^^^^^^

The spheroidal model of order 3.

.. math::
   C(r) = \begin{cases}
     1 - ar & r \leq r_0 \\
     b (1 + c r^2)^{-3/2} & \mathrm{otherwise}
    \end{cases}

Parameters :math:`r_0, a, b` and `c` are chosen so that the function will be smooth and :math:`C(1) \simeq 0.037`. :math:`C(\|\cdot\|)` is positive definite on :math:`\mathbb{R}^d` up to :math:`d = 3` at least (checked numerically).

The spheroidal model is a piecewise function consists of the linear and the Cauchy model. It approximates the spherical model while has smoother shape (:math:`C(r)` is of class :math:`C^2` on :math:`(0, \infty)`). It does not have a geometrical interpretation as the spherical model does. See an `introduction by Seequent Limited <https://blog.leapfrog3d.com/2019/06/11/the-spheroidal-family-of-variograms-explained/>`_ for details.

cov_spheroidal5
^^^^^^^^^^^^^^^

The spheroidal model of order 5.

.. math::
   C(r) = \begin{cases}
     1 - ar & r \leq r_0 \\
     b (1 + c r^2)^{-5/2} & \mathrm{otherwise}
    \end{cases}

:math:`C(\|\cdot\|)` is positive definite on :math:`\mathbb{R}^d` up to :math:`d = 3` at least (checked numerically).

cov_spheroidal7
^^^^^^^^^^^^^^^

The spheroidal model of order 7.

.. math::
   C(r) = \begin{cases}
     1 - ar & r \leq r_0 \\
     b (1 + c r^2)^{-7/2} & \mathrm{otherwise}
    \end{cases}

:math:`C(\|\cdot\|)` is positive definite on :math:`\mathbb{R}^d` up to :math:`d = 3` at least (checked numerically).

cov_spheroidal9
^^^^^^^^^^^^^^^

The spheroidal model of order 9.

.. math::
   C(r) = \begin{cases}
     1 - ar & r \leq r_0 \\
     b (1 + c r^2)^{-9/2} & \mathrm{otherwise}
    \end{cases}

:math:`C(\|\cdot\|)` is positive definite on :math:`\mathbb{R}^d` up to :math:`d = 3` at least (checked numerically).

The nugget effect model
-----------------------

The nugget effect model is not implemented as a RBF. Instead, you can add it by calling ``model::set_nugget`` function. The nugget effect model is included in fitting but excluded from evaluation to keep the interpolant continuous.

.. math::
   C(r) = \begin{cases}
     1 & r = 0 \\
     0 & \mathrm{otherwise}
    \end{cases}

:math:`C(\|\cdot\|)` is positive definite on every :math:`\mathbb{R}^d`.
