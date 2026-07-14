# Structural LVA prototype

The `feature/structural-lva` branch contains a native Polatory structural-LVA path.

## Python API

```python
import numpy as np
import polatory
from polatory import three as p3

rbf = p3.CovSpheroidal3([10.0, 100.0])
model = p3.Model(rbf, 0)

trend_input = polatory.StructuralTrendInput3(
    vertices=mesh_vertices,
    faces=mesh_faces.astype(np.int64),
    strength=5.0,
    range=50.0,
)

structural = polatory.StructuralInterpolant3(
    model,
    outside_value=-1.0,
    blend_power=7.0,
)

domains = structural.fit_from_meshes(
    points,
    values,
    [trend_input],
    tolerance=1e-6,
    trend_type=polatory.StructuralTrendType.STRONGEST_ALONG_INPUTS,
)

predictions = structural.evaluate(query_points)
```

`fit_from_meshes` automatically:

1. calculates equal-weight incident triangle normals at mesh vertices;
2. finds the nearest mesh vertex to each structural sample location;
3. applies the recovered single-input decay law;
4. constructs determinant-one local anisotropy matrices;
5. divides the interpolation data into overlapping spatial domains;
6. assigns every data point inside each expanded domain box as local support;
7. fits and blends the local Polatory interpolants.

## Recovered single-input field

For the nearest mesh-vertex normal `n`, distance `d`, input `strength`, and input `range`:

```text
q = exp(-d / range)
r = 1 + (strength - 1) * q
M = r^(-1/3) * (I - n n^T) + r^(2/3) * (n n^T)
```

The equal-weight vertex normal and this matrix equation were verified against the supplied Leapfrog project.

## Trend types

The public API exposes Leapfrog-style names:

- `STRONGEST_ALONG_INPUTS`
- `BLENDING`
- `NON_DECAYING`

Multiple mesh inputs are accepted. `STRONGEST_ALONG_INPUTS` selects the input with the largest local decayed anisotropy contribution.

`BLENDING` currently uses an axial weighted-normal blend. This is an experimental implementation. The Leapfrog manual confirms that multiple inputs are blended according to individual strength and that the result decays away from the meshes, but it does not publish the exact vector/tensor combination rule.

## Parameters suitable for an RSGeo UI

User-facing structural trend settings:

- name;
- trend type;
- one or more mesh inputs;
- per-input strength;
- per-input range for decaying modes;
- optional global mean trend in a later parity update;
- compatibility mode in a later parity update.

Advanced internal settings that should normally stay hidden:

- domain size;
- domain overlap;
- minimum local support count;
- local blend power.

## Known parity limits

The following still require controlled Leapfrog A/B tests before claiming exact parity:

1. multiple-input `BLENDING` orientation and strength equations;
2. transition behaviour where two inputs have equal influence in `STRONGEST_ALONG_INPUTS`;
3. global mean trend interaction;
4. Version 1 versus Version 2 compatibility;
5. Leapfrog's internal automatic domain decomposition;
6. whether one tuned local blend power generalises across unrelated datasets.

The automatic domain builder is therefore ready for compilation and single-mesh validation, while multi-mesh blending remains explicitly experimental.
