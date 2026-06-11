(theory-staged-analysis)=
# Staged (evolutive) sectional analysis

Some sections are built and loaded in stages: parts are added (activated) at
different moments and each part only feels the deformation that occurs *after*
it becomes active. The typical example is a steel-concrete composite section:
loads are first carried by the bare steel part and, once the concrete slab and
its reinforcement are activated, additional loads are shared by the whole
section while the steel keeps its already accumulated deformation.

## Reference strain plane

Each part of the section is given a **stress-free reference strain plane**
$P_{ref} = (\varepsilon_a, \chi_y, \chi_z)$: the section strain plane that
existed when the part became active. The strain that drives the constitutive
law of each fiber is then the difference between the current plane and the
reference plane of its part:

:::{math}
:label: eq:staged-driving-strain
\varepsilon_{drive}(y, z) = P_{current}(y, z) - P_{ref}^{part}(y, z)
:::

where, consistently with {eq}`eq:fiber_strain_fibers`, a plane evaluated at a
point is $\varepsilon_a + \chi_y \cdot z - \chi_z \cdot y$.

* Parts active from the beginning (the steel) have $P_{ref} = 0$ and feel the
  full plane.
* Parts activated later (the slab, the reinforcement) feel only the increment
  $P_{current} - P_{ref}$ since their activation, so they are stress-free under
  the loads already present when they were activated.

Equilibrium is solved for the cumulative total loads, summing the stresses of
the active parts only, each one with its own reference plane.

## Not the same as material prestrain

This reference plane must not be confused with the material
`initial_strain` / `InitialStrain` used for prestressing:

```{list-table}
:header-rows: 1

* - 
  - Material prestrain (`initial_strain`)
  - Reference strain plane (staged)
* - Nature
  - Property of the material
  - State of a geometry instance in time
* - Spatial variation
  - Scalar, uniform over the material
  - Plane $(\varepsilon_a, \chi_y, \chi_z)$, varies linearly with $y, z$
* - Where it lives
  - Wrapped in the constitutive law
  - In the analysis context (not in the geometry)
* - Use
  - Prestressing / pretensioning
  - Construction sequence, composite sections
```

The two are orthogonal: a slab could have both a staged reference plane and
prestressed reinforcement.

## Implementation notes

* The predeformed state lives in a `SectionStrainState` owned by the
  `StagedSectionCalculator`, **not** on the `Geometry` objects, which stay
  stateless and reusable.
* Solving a load case is strictly read-only on the state: the solved plane is
  returned in a result object and never written back. Consequently, evaluating
  several envelope load cases on the same baseline (e.g. $M_{y,\max}$ and
  $M_{y,\min}$) is independent by construction and cannot accumulate state.
  Use `reset()` to return to the virgin model and `snapshot()` / `restore()` to
  branch from a common locked baseline.
* Only the `fiber` integrator supports staged analyses. The `marin` integrator
  rotates the whole geometry to a single uniaxial plane, which is incompatible
  with per-part reference planes. With the fiber integrator the reference
  strain is precomputed per fiber as a scalar, so it is invariant to rotation.
