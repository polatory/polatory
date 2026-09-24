#!/usr/bin/env python3

import gc
import os
import tempfile

import numpy as np
import polatory
import polatory.one
import polatory.three
import polatory.two

MODULES = [(1, polatory.one), (2, polatory.two), (3, polatory.three)]

RBF_NAMES = [
    "Biharmonic2D",
    "Biharmonic3D",
    "CovCubic",
    "CovExponential",
    "CovGaussian",
    "CovGeneralizedCauchy3",
    "CovGeneralizedCauchy5",
    "CovGeneralizedCauchy7",
    "CovGeneralizedCauchy9",
    "CovSpherical",
    "CovSpheroidal3",
    "CovSpheroidal5",
    "CovSpheroidal7",
    "CovSpheroidal9",
    "Triharmonic2D",
    "Triharmonic3D",
]

rng = np.random.default_rng(0)


def assert_raises(exc_type, fn, *args):
    try:
        fn(*args)
    except exc_type:
        return
    raise AssertionError(f"{exc_type.__name__} not raised")


def min_pairwise_distance(points):
    d = np.linalg.norm(points[:, None] - points[None], axis=2)
    np.fill_diagonal(d, np.inf)
    return d.min()


def test_bbox(dim, mod):
    points = rng.random((100, dim))
    bbox = mod.Bbox.from_points(points)
    assert not bbox.is_empty
    assert np.array_equal(bbox.min, points.min(axis=0))
    assert np.array_equal(bbox.max, points.max(axis=0))
    assert mod.Bbox().is_empty

    bbox = mod.Bbox(np.zeros(dim), np.ones(dim))
    assert np.array_equal(bbox.max, np.ones(dim))
    assert_raises(TypeError, mod.Bbox, np.zeros((2, dim)), np.ones((2, dim)))


def test_rbfs(dim, mod):
    diff = np.full(dim, 0.1)
    for name in RBF_NAMES:
        params = [1.0] if name.endswith(("2D", "3D")) else [1.0, 0.5]
        rbf = getattr(mod, name)(params)
        assert rbf.parameters[: len(params)] == params
        assert rbf.num_parameters == len(rbf.parameter_names)
        assert rbf.num_parameters == len(rbf.parameter_lower_bounds)
        assert rbf.num_parameters == len(rbf.parameter_upper_bounds)
        assert rbf.is_covariance_function == name.startswith("Cov")
        assert rbf.cpd_order >= 0
        assert rbf.short_name

        rbf.evaluate(diff)
        rbf.evaluate_gradient(diff)
        try:
            rbf.evaluate_hessian(diff)
        except RuntimeError as e:
            assert "not implemented" in str(e)

        aniso = rbf.anisotropy
        aniso.flat[0] = 2.0
        assert rbf.anisotropy.flat[0] == 1.0
        rbf.anisotropy = aniso
        assert rbf.anisotropy.flat[0] == 2.0

    assert_raises(ValueError, mod.CovExponential, [1.0])


def test_model(dim, mod, tmp):
    model = mod.Model([mod.CovExponential([1.0, 0.5]), mod.CovGaussian([2.0, 0.7])])
    model.nugget = 0.1
    assert model.is_covariance_model
    assert model.num_rbfs == 2
    assert model.parameters == [0.1, 1.0, 0.5, 2.0, 0.7]
    assert model.parameter_names == ["nugget", "psill", "range", "psill", "range"]
    assert len(model.parameter_lower_bounds) == model.num_parameters
    assert len(model.parameter_upper_bounds) == model.num_parameters
    assert model.description

    model.rbfs[0].parameters = [3.0, 0.9]
    assert model.parameters == [0.1, 1.0, 0.5, 2.0, 0.7]

    path = os.path.join(tmp, f"model{dim}")
    model.save(path)
    assert mod.Model.load(path).parameters == model.parameters

    model = mod.Model([mod.CovExponential([1.0, 0.5]), mod.Biharmonic3D([1.0])])
    assert not model.is_covariance_model
    assert_raises(RuntimeError, lambda: model.description)

    model = mod.Model(mod.Biharmonic3D([1.0]), poly_degree=1)
    assert model.poly_degree == 1
    assert model.poly_basis_size > 0
    assert model.cpd_order == 1
    assert mod.Model.MIN_REQUIRED_POLY_DEGREE == -2


def test_interpolant(dim, mod, tmp):
    points = rng.random((200, dim))
    values = np.sin(3.0 * points.sum(axis=1))
    grad_points = points[:10]
    grad_values = np.repeat(3.0 * np.cos(3.0 * grad_points.sum(axis=1)), dim)
    all_values = np.concatenate([values, grad_values])

    # Biharmonic3D reduces to |x| in 1D, which cannot fit gradients.
    interp = mod.Interpolant(mod.Model(mod.Triharmonic3D([1.0]), poly_degree=1))
    interp.fit(points, values, 1e-6)
    assert np.abs(interp.evaluate(points) - values).max() < 1e-5
    assert interp.evaluate(points, grad_points).shape == all_values.shape

    weights = interp.weights
    weights[0] += 1.0
    assert interp.weights[0] != weights[0]
    centers = interp.centers
    centers.flat[0] += 1.0
    assert interp.centers.flat[0] != centers.flat[0]
    interp.model.nugget = 1.0
    assert interp.model.nugget == 0.0
    assert interp.grad_centers.size == 0
    assert np.array_equal(interp.bbox.min, points.min(axis=0))

    interp.fit(points, grad_points, all_values, 1e-6, 1e-6, initial=interp)
    assert np.abs(interp.evaluate(points, grad_points) - all_values).max() < 1e-5
    interp.fit_incrementally(points, values, 1e-6)
    interp.fit_incrementally(points, grad_points, all_values, 1e-6, 1e-6)
    nan = np.full(len(values), np.nan)
    interp.fit_inequality(points, values, nan, nan, 1e-6)

    path = os.path.join(tmp, f"interpolant{dim}")
    interp.save(path)
    assert np.array_equal(mod.Interpolant.load(path).weights, interp.weights)


def test_distance_filter(dim, mod):
    points = rng.random((1000, dim))
    distance = 0.05 if dim > 1 else 0.001

    f = mod.DistanceFilter(points)
    indices = f.filter(distance).filtered_indices
    assert 0 < len(indices) <= len(points)
    assert min_pairwise_distance(points[indices]) >= distance

    subset = list(range(0, len(points), 2))
    indices = f.filter(distance, subset).filtered_indices
    assert set(indices) <= set(subset)
    assert min_pairwise_distance(points[indices]) >= distance

    assert_raises(ValueError, f.filter, distance, [0, len(points)])
    assert_raises(ValueError, f.filter, distance, [-1])


def test_kriging(dim, mod, tmp):
    points = rng.random((200, dim))
    values = np.sin(3.0 * points.sum(axis=1))

    calc = mod.VariogramCalculator(0.1, 5)
    calc.angle_tolerance = mod.VariogramCalculator.AUTOMATIC_ANGLE_TOLERANCE
    calc.lag_tolerance = mod.VariogramCalculator.AUTOMATIC_LAG_TOLERANCE
    calc.directions = mod.VariogramCalculator.ANISOTROPIC_DIRECTIONS
    directions = calc.directions
    expected = directions.copy()
    calc.directions = mod.VariogramCalculator.ISOTROPIC_DIRECTIONS
    assert np.array_equal(directions, expected)

    variog_set = calc.calculate(points, values)
    assert variog_set.num_variograms == 1
    variog = variog_set.variograms[0]
    assert len(variog.bin_distance) == variog.num_bins
    assert len(variog.bin_gamma) == variog.num_bins
    assert sum(variog.bin_num_pairs) == variog.num_pairs
    assert variog.direction.shape == (dim,)

    path = os.path.join(tmp, f"variog_set{dim}")
    variog_set.save(path)
    assert mod.VariogramSet.load(path).num_pairs == variog_set.num_pairs

    fit = mod.VariogramFitting(
        variog_set,
        mod.Model(mod.CovExponential([1.0, 0.5]), poly_degree=-1),
        polatory.WeightFunction.NUM_PAIRS,
        fit_anisotropy=dim > 1,
    )
    assert fit.brief_report and fit.full_report
    assert fit.final_cost >= 0.0
    assert fit.model.num_rbfs == 1

    nst = polatory.NormalScoreTransformation()
    y = nst.transform(values)
    assert abs(y.mean()) < 1e-6 and abs(y.std() - 1.0) < 1e-2
    assert np.corrcoef(nst.back_transform(y), values)[0, 1] > 0.99
    variog_set.back_transform(nst)
    variog.back_transform(nst)

    set_ids = np.arange(len(points)) % 5
    model = mod.Model(mod.CovExponential([1.0, 0.5]), poly_degree=0)
    assert (
        mod.cross_validate(model, points, values, set_ids, 1e-6).shape == values.shape
    )
    assert mod.detrend(points, values, 1).shape == values.shape


def sphere_points(n):
    points = rng.normal(size=(n, 3))
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def test_normal_estimator():
    points = sphere_points(500)
    up = np.array([0.0, 0.0, 1.0])

    ne = polatory.NormalEstimator(points)
    assert ne.estimate_with_knn(20) is ne
    ne.orient_toward_direction(up)
    assert (ne.normals @ up).min() >= 0.0
    ne.estimate_with_knn([10, 20]).filter_by_plane_factor().orient_closed_surface()
    ne.estimate_with_radius(0.3).orient_toward_point(np.zeros(3))
    ne.estimate_with_radius([0.2, 0.3]).filter_by_plane_factor(
        1.8
    ).orient_closed_surface(50)
    assert ne.normals.shape == (500, 3)
    assert ne.plane_factors.shape == (500,)

    normals = ne.normals
    expected = normals.copy()
    normals[:] = 7.0
    assert np.array_equal(ne.normals, expected)

    sdf = polatory.SdfDataGenerator(points, ne.normals, 0.1)
    assert len(sdf.sdf_points) == len(sdf.sdf_values)
    sdf = polatory.SdfDataGenerator(points, ne.normals, 0.1, np.eye(3))
    assert len(sdf.sdf_points) == len(sdf.sdf_values)


def fit_plane_interpolant():
    points = rng.random((200, 3))
    interp = polatory.three.Interpolant(
        polatory.three.Model(polatory.three.Biharmonic3D([1.0]), poly_degree=1)
    )
    interp.fit(points, points[:, 0] - 0.5, 1e-6)
    return interp


def test_isosurface(tmp):
    bbox = polatory.three.Bbox(np.zeros(3), np.ones(3))
    interp = fit_plane_interpolant()
    field_fn = polatory.RbfFieldFunction(interp)

    mesh = polatory.Isosurface(bbox, 0.1).generate(field_fn)
    assert mesh.faces.shape[0] > 0
    assert not mesh.is_empty and not mesh.is_entire
    assert np.abs(mesh.vertices[:, 0] - 0.5).max() < 1e-6
    mesh.export_obj(os.path.join(tmp, "mesh.obj"))

    mesh = polatory.Isosurface(bbox, 0.1).generate(field_fn, isovalue=1.0)
    assert mesh.is_entire and not mesh.is_empty and mesh.faces.shape[0] == 0
    mesh = polatory.Isosurface(bbox, 0.1).generate(field_fn, isovalue=-1.0)
    assert mesh.is_empty and not mesh.is_entire and mesh.faces.shape[0] == 0

    iso = polatory.Isosurface(bbox, 0.1, np.eye(3))
    snap_points = np.column_stack([np.full(5, 0.5), rng.random((5, 2))])
    iso.set_snap_points(snap_points)
    iso.set_snap_points(snap_points, np.full(5, 0.5))
    seed_points = np.array([[0.5, 0.5, 0.5]])
    field_fn = polatory.RbfFieldFunction(interp, 1e-6, 1e-5)
    mesh = iso.generate_from_seed_points(
        seed_points, field_fn, isovalue=0.0, refine=False
    )
    assert mesh.faces.shape[0] > 0


def test_isosurface_25d():
    points = rng.random((100, 2))
    interp = polatory.two.Interpolant(
        polatory.two.Model(polatory.two.Biharmonic2D([1.0]))
    )
    interp.fit(points, 0.1 * points.sum(axis=1), 1e-6)

    bbox = polatory.three.Bbox(np.array([0.0, 0.0, -1.0]), np.array([1.0, 1.0, 1.0]))
    mesh = polatory.Isosurface(bbox, 0.1).generate(polatory.RbfFieldFunction25D(interp))
    assert mesh.faces.shape[0] > 0


def test_field_function_keeps_interpolant_alive():
    field_fn = polatory.RbfFieldFunction(fit_plane_interpolant())
    gc.collect()
    model = polatory.three.Model(polatory.three.Biharmonic3D([1.0]))
    garbage = [polatory.three.Interpolant(model) for i in range(1000)]

    bbox = polatory.three.Bbox(np.zeros(3), np.ones(3))
    mesh = polatory.Isosurface(bbox, 0.1).generate(field_fn)
    assert np.abs(mesh.vertices[:, 0] - 0.5).max() < 1e-6


def test_weight_function():
    polatory.WeightFunction(1.0, 0.0, 1.0)
    for name in [
        "NUM_PAIRS",
        "NUM_PAIRS_OVER_DISTANCE_SQUARED",
        "NUM_PAIRS_OVER_MODEL_GAMMA_SQUARED",
        "ONE",
        "ONE_OVER_DISTANCE_SQUARED",
        "ONE_OVER_MODEL_GAMMA_SQUARED",
    ]:
        assert isinstance(
            getattr(polatory.WeightFunction, name), polatory.WeightFunction
        )


def main():
    with tempfile.TemporaryDirectory() as tmp:
        for dim, mod in MODULES:
            test_bbox(dim, mod)
            test_rbfs(dim, mod)
            test_model(dim, mod, tmp)
            test_interpolant(dim, mod, tmp)
            test_distance_filter(dim, mod)
            test_kriging(dim, mod, tmp)
        test_normal_estimator()
        test_isosurface(tmp)
        test_isosurface_25d()
        test_field_function_keeps_interpolant_alive()
        test_weight_function()
    print("All tests passed.")


if __name__ == "__main__":
    main()
