#!/usr/bin/env python3

import numpy as np

import polatory as p
import polatory.three as p3


def main() -> None:
    # データの読み込み
    data = np.loadtxt("data/grade.csv", delimiter=",")
    points, values = data[:, :3], data[:, 3]

    # 経験バリオグラムの計算
    emp_variog = p3.EmpiricalVariogram(points, values, bin_width=5.0, num_bins=100)

    # モデルの定義 (球状型モデル × 2 + ナゲット効果モデル)
    cov1 = p3.CovSpheroidal5([0.3, 75.0])
    cov2 = p3.CovSpheroidal5([0.3, 75.0])
    m = p3.Model([cov1, cov2], poly_degree=0)

    # モデルを経験バリオグラムにフィッティング
    fit = p3.VariogramFitting(emp_variog, m)
    print(fit.parameters)
    m.parameters = fit.parameters

    # 補間
    inter = p3.Interpolant(m)
    inter.fit(points, values, absolute_tolerance=1e-4)

    # 等値面の生成
    fn = p.RbfFieldFunction(inter)
    bbox = p3.Bbox.from_points(points)
    iso = p.Isosurface(bbox, resolution=5.0)
    surf = iso.generate(fn, isovalue=1.0)
    surf.export_obj("data/grade.obj")


if __name__ == "__main__":
    main()
