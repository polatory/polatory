#!/usr/bin/env python3

import numpy as np

data = np.loadtxt("data/sphere.asc")
for v in data:
    x, y, z = v
    if z < 0.0 or z == 0 and y < 0 or z == 0 and y == 0 and x < 0:
        continue
    # C++
    # print("{" + ",".join(str(x) for x in v) + "},")
    # Python
    print("[" + ",".join(str(x) for x in v) + "],")
