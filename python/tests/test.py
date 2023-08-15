import numpy as np
import numpy.testing as npt
import polatory as po


def test():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    values = np.array([0.0, 1.0, 1.0, 1.0])
    rbf = po.Biharmonic3D([1.0])
    m = po.Model(rbf, 3, 1)
    inter = po.Interpolant(m)
    inter.fit(points, values, 1e-10)

    npt.assert_almost_equal(
        inter.evaluate(np.array([[0.5, 0.5, 0.5]])), np.array([1.5]), decimal=10
    )
