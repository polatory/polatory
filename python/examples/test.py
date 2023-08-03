#!/usr/bin/env python3

import numpy as np
import polatory as po

horse = np.loadtxt("../../data/horse.asc", delimiter=",")
points, normals = horse[:, :3], horse[:, 3:]

sdf = po.SdfDataGenerator(points, normals, 1e-4, 1e-3)
sdf_points, sdf_values = sdf.sdf_points, sdf.sdf_values

idcs = po.DistanceFilter(sdf_points, 1e-10).filtered_indices
sdf_points, sdf_values = sdf_points[idcs, :], sdf_values[idcs]

rbf = po.Biharmonic3D([1.0])
m = po.Model(rbf, 3, 0)
inter = po.Interpolant(m)
inter.fit(sdf_points, sdf_values, 5e-5)

# print("values:", inter.evaluate(points))
# print("centers:", inter.centers)
# print("weights:", inter.weights)

bbox = po.Bbox3d([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
fn = po.RbfFieldFunction(inter)
iso = po.Isosurface(bbox, 5e-4)
surf = iso.generate_from_seed_points(points, fn)
surf.export_obj("horse.obj")
