#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

import polatory as p
import polatory.three as p3


def main() -> None:
    data = np.loadtxt("data/grade.csv", delimiter=",")
    points, values = data[:, :3], data[:, 3]

    values = p3.detrend(points, values, 2)

    sph_variog = p.SphericalVariogram(points, values, bin_width=5.0, num_bins=20)
    variogs = sph_variog.into_variograms()
    for v in variogs:
        d, g, n = v.bin_distance, v.bin_gamma, v.bin_num_pairs
        for d, g, n in zip(d, g, n):
            print(f"{d}, {g}, {n}")
        print("----")

    cov1 = p3.CovSpherical([0.5, 30.0])
    covs = [cov1]
    m = p3.Model(covs, poly_degree=0)

    for _ in range(50):
        fit = p.VarioFitting(variogs, m)

        ea = fit.euler_angles * 180 / np.pi
        pitch, dip, dip_az = ea
        if dip < -90.0:
            dip += 180.0
            pitch = 180.0 - pitch
        elif dip < 0.0:
            dip = -dip
            dip_az += 180.0
        elif dip > 90.0:
            dip = 180.0 - dip
            dip_az += 180.0
            pitch = 180.0 - pitch
        if dip_az < 0.0:
            dip_az += 360.0
        elif dip_az > 360.0:
            dip_az -= 360.0
        print("         dip      dip az       pitch")
        print(f"  {dip:10.4f}  {dip_az:10.4f}  {pitch:10.4f}")

        print("       psill         maj         mid         min")
        nug = fit.parameters[0]
        print(f"  {nug:10.4f}")
        for i in range(len(covs)):
            psill = fit.parameters[1 + 2 * i]
            r = fit.parameters[1 + 2 * i + 1]
            maj = r / fit.scale(i)[0]
            mid = r / fit.scale(i)[1]
            min = r / fit.scale(i)[2]
            print(f"  {psill:10.4f}  {maj:10.4f}  {mid:10.4f}  {min:10.4f}")


if __name__ == "__main__":
    main()
