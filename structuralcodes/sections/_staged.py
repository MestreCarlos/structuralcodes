"""Staged (evolutive) sectional analysis.

This module implements a staged sectional analysis on top of the existing
``GenericSection`` machinery. Parts of a section (typically the
``extra_geometries`` of a composite section, e.g. the concrete slab and its
reinforcement of a steel-concrete section) can be activated/deactivated and
each part carries a *stress-free reference strain plane*: the section strain
plane that existed when the part became active.

The strain that drives the constitutive law of each fiber becomes::

    eps_drive(y, z) = P_current(y, z) - P_reference_part(y, z)

so parts activated later (the slab) only feel the strain increment since their
activation, while parts active from the beginning (the steel) feel the full
plane.

Design decisions (see plan):

* The predeformed state does **not** live on the ``Geometry`` objects, which
  stay stateless and reusable. It lives in a ``SectionStrainState`` owned by
  the ``StagedSectionCalculator``.
* Solving a load case is strictly **read-only** on the state: it never writes
  the solved plane back. Only the staging (``run`` / ``freeze_new_active``)
  writes reference planes, and it writes them to the *state*, not the geometry.
  Therefore two envelope load cases on the same baseline are independent by
  construction and cannot accumulate state.

This is conceptually orthogonal to the material ``initial_strain`` /
``InitialStrain`` (a scalar baked in the constitutive law, used for
prestressing): here the reference is a *plane* attached to a geometry instance.

Only the ``fiber`` integrator is supported for staged analyses (the ``marin``
integrator rotates the whole geometry to a single uniaxial plane, which is
incompatible with per-part reference planes).
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

from structuralcodes.geometry import (
    CompoundGeometry,
    Geometry,
    SurfaceGeometry,
)

from ._generic import GenericSection

# A strain plane is the triplet (eps_a, chi_y, chi_z).
StrainPlane = t.Tuple[float, float, float]
# Anything that can be resolved to a set of geometries.
GeometrySelector = t.Union[Geometry, str, t.Iterable[t.Union[Geometry, str]]]


class SectionStrainState:
    """Holds the evolutive state of a staged sectional analysis.

    The state is a pure data container, independent from the geometry objects.
    It keeps track of which geometries are active and of the stress-free
    reference strain plane of each part.

    Attributes:
        all_geometries (List(Geometry)): All geometries known by the state, in
            a stable order.
    """

    def __init__(
        self, geometries: t.Optional[t.Iterable[Geometry]] = None
    ) -> None:
        """Initialize the state.

        Arguments:
            geometries (Optional(Iterable(Geometry))): All geometries handled
                by this state. By default all of them are active and none has
                a reference plane (virgin model).
        """
        self._all_geometries: t.List[Geometry] = (
            list(geometries) if geometries is not None else []
        )
        self._reference_planes: t.Dict[Geometry, StrainPlane] = {}
        self._active: t.Set[Geometry] = set(self._all_geometries)

    @property
    def all_geometries(self) -> t.List[Geometry]:
        """Return all geometries known by the state (stable order)."""
        return list(self._all_geometries)

    @property
    def active_geometries(self) -> t.List[Geometry]:
        """Return the active geometries, preserving the original order."""
        return [g for g in self._all_geometries if g in self._active]

    def is_active(self, geometry: Geometry) -> bool:
        """Return True if the geometry is currently active."""
        return geometry in self._active

    def activate(self, geometry: Geometry) -> None:
        """Mark a geometry as active."""
        self._active.add(geometry)
        if geometry not in self._all_geometries:
            self._all_geometries.append(geometry)

    def deactivate(self, geometry: Geometry) -> None:
        """Mark a geometry as inactive."""
        self._active.discard(geometry)

    def activate_all(self) -> None:
        """Activate every known geometry."""
        self._active = set(self._all_geometries)

    def deactivate_all(self) -> None:
        """Deactivate every geometry."""
        self._active = set()

    def reference_plane(self, geometry: Geometry) -> t.Optional[StrainPlane]:
        """Return the reference strain plane of a geometry (None if null)."""
        return self._reference_planes.get(geometry)

    def set_reference_plane(
        self, geometry: Geometry, plane: StrainPlane
    ) -> None:
        """Set the stress-free reference strain plane of a geometry."""
        self._reference_planes[geometry] = (
            float(plane[0]),
            float(plane[1]),
            float(plane[2]),
        )

    def clear_reference_plane(self, geometry: Geometry) -> None:
        """Remove the reference strain plane of a geometry (back to null)."""
        self._reference_planes.pop(geometry, None)

    def clear_all_references(self) -> None:
        """Remove the reference strain plane of every geometry."""
        self._reference_planes.clear()

    @property
    def reference_planes(self) -> t.Dict[Geometry, StrainPlane]:
        """Return a copy of the geometry -> reference plane mapping."""
        return dict(self._reference_planes)

    @property
    def has_predeformation(self) -> bool:
        """Return True if any geometry carries a reference strain plane."""
        return len(self._reference_planes) > 0

    def reset(self) -> None:
        """Reset to the virgin model: no references and all parts active."""
        self.clear_all_references()
        self.activate_all()

    def snapshot(self) -> SectionStrainState:
        """Return an independent copy of the current state.

        The copy shares the geometry instances but has its own reference
        planes and active set, so it can be used as a baseline to branch
        several envelope load cases.
        """
        new = SectionStrainState(self._all_geometries)
        new._reference_planes = dict(self._reference_planes)
        new._active = set(self._active)
        return new

    def restore(self, snapshot: SectionStrainState) -> None:
        """Restore the state from a previously taken snapshot."""
        self._all_geometries = list(snapshot._all_geometries)
        self._reference_planes = dict(snapshot._reference_planes)
        self._active = set(snapshot._active)


@dataclass
class StagedAnalysisResult:
    """Result of a single staged load-case solve.

    Attributes:
        eps_a (float): Axial strain of the solved plane at (0, 0).
        chi_y (float): Curvature around the y-axis.
        chi_z (float): Curvature around the z-axis.
        n (float): Applied axial load.
        my (float): Applied bending moment around y.
        mz (float): Applied bending moment around z.
        section (GenericSection): The (active subset) section used to solve.
        reference_planes (Dict): The reference planes applied while solving.
        active_geometries (List(Geometry)): The active geometries.
    """

    eps_a: float
    chi_y: float
    chi_z: float
    n: float
    my: float
    mz: float
    section: GenericSection
    reference_planes: t.Dict[Geometry, StrainPlane] = field(
        default_factory=dict
    )
    active_geometries: t.List[Geometry] = field(default_factory=list)

    @property
    def strain_plane(self) -> StrainPlane:
        """Return the solved strain plane as a tuple (eps_a, chi_y, chi_z)."""
        return (self.eps_a, self.chi_y, self.chi_z)

    def driving_strain(
        self, y: float, z: float, geometry: t.Optional[Geometry] = None
    ) -> float:
        """Return the strain that drives the constitutive law at a point.

        It is ``P_current(y, z) - P_reference(y, z)`` where the reference plane
        is the one of ``geometry`` (null if not given or absent).

        Arguments:
            y (float): Section Y coordinate of the point.
            z (float): Section Z coordinate of the point.
            geometry (Optional(Geometry)): The part the point belongs to. When
                given, its reference plane is subtracted.

        Returns:
            float: The driving strain at the point.
        """
        eps = self.eps_a + self.chi_y * z - self.chi_z * y
        if geometry is not None:
            ref = self.reference_planes.get(geometry)
            if ref is not None:
                eps -= ref[0] + ref[1] * z - ref[2] * y
        return eps

    def stress_resultants(self) -> t.Tuple[float, float, float]:
        """Integrate and return the stress resultants (N, My, Mz).

        Useful to verify equilibrium with the applied loads.
        """
        return tuple(
            self.section.section_calculator.integrate_strain_profile(
                strain=list(self.strain_plane)
            )
        )


class StagedSectionCalculator:
    """Orchestrator for staged (evolutive) sectional analysis.

    It owns the full geometry and a ``SectionStrainState``. Stages are defined
    with ``add_stage`` and executed with ``run``; ``solve`` evaluates a single
    load case on the current active subset in a strictly read-only fashion.

    Attributes:
        geometry (CompoundGeometry): The full section geometry.
        state (SectionStrainState): The evolutive state.
        mesh_size (float): The mesh size for the fiber integrator.
    """

    def __init__(
        self,
        geometry: t.Union[SurfaceGeometry, CompoundGeometry],
        mesh_size: float = 0.01,
    ) -> None:
        """Initialize the staged calculator.

        Arguments:
            geometry (Union(SurfaceGeometry, CompoundGeometry)): The full
                section geometry (all parts that can be activated).
            mesh_size (float): Mesh size (0 to 1) for the fiber integrator
                (default 0.01).
        """
        if isinstance(geometry, SurfaceGeometry):
            geometry = CompoundGeometry([geometry])
        self.geometry = geometry
        self.mesh_size = mesh_size
        all_geometries: t.List[Geometry] = list(geometry.geometries) + list(
            geometry.point_geometries
        )
        self.state = SectionStrainState(all_geometries)
        self._stages: t.List[t.Tuple[GeometrySelector, StrainPlane]] = []

    # -- state lifecycle (delegated) -------------------------------------
    @property
    def has_predeformation(self) -> bool:
        """Return True if any part carries a reference strain plane."""
        return self.state.has_predeformation

    def reset(self) -> None:
        """Reset to the virgin model (no references, all parts active)."""
        self.state.reset()

    def snapshot(self) -> SectionStrainState:
        """Return an independent snapshot of the current state."""
        return self.state.snapshot()

    def restore(self, snapshot: SectionStrainState) -> None:
        """Restore the state from a snapshot."""
        self.state.restore(snapshot)

    # -- geometry selection ----------------------------------------------
    def _resolve_geometries(
        self, selector: GeometrySelector
    ) -> t.List[Geometry]:
        """Resolve a selector to a list of known geometry instances.

        The selector can be a single ``Geometry``, a ``group_label`` string, or
        an iterable mixing both.
        """
        resolved: t.List[Geometry] = []
        if isinstance(selector, Geometry):
            resolved.append(selector)
        elif isinstance(selector, str):
            resolved.extend(
                g
                for g in self.state.all_geometries
                if g.group_label == selector
            )
        else:
            for item in selector:
                resolved.extend(self._resolve_geometries(item))
        # Keep unique, preserving order.
        seen: t.List[Geometry] = []
        for g in resolved:
            if g not in seen:
                seen.append(g)
        return seen

    # -- staging ----------------------------------------------------------
    def add_stage(
        self,
        geometries: GeometrySelector,
        n: float = 0.0,
        my: float = 0.0,
        mz: float = 0.0,
    ) -> None:
        """Add a stage to the construction sequence.

        Arguments:
            geometries (GeometrySelector): The geometries (or group labels)
                that are active in this stage. Parts activated in previous
                stages remain active.
            n (float): Total axial load at this stage (cumulative).
            my (float): Total bending moment around y at this stage.
            mz (float): Total bending moment around z at this stage.
        """
        self._stages.append((geometries, (n, my, mz)))

    def freeze_new_active(
        self,
        geometries: t.Iterable[Geometry],
        plane: t.Optional[StrainPlane],
    ) -> None:
        """Freeze the reference strain plane of newly activated parts.

        Activates the given geometries and, when a plane is provided, records
        it as their stress-free reference (only for parts that were not active
        before). This is the only write to the state.

        Arguments:
            geometries (Iterable(Geometry)): The geometries to activate.
            plane (Optional(StrainPlane)): The current section strain plane to
                freeze as reference. None for the very first stage.
        """
        for g in geometries:
            newly_active = not self.state.is_active(g)
            self.state.activate(g)
            if newly_active and plane is not None:
                self.state.set_reference_plane(g, plane)

    # -- solving (read-only) ---------------------------------------------
    def _build_active_section(self) -> GenericSection:
        """Build a ``GenericSection`` for the current active subset.

        The fiber integrator of the new section is configured with the current
        reference planes. This does not mutate the state.
        """
        active = self.state.active_geometries
        if not active:
            raise ValueError('No active geometries to solve the section.')
        section = GenericSection(
            CompoundGeometry(active),
            integrator='fiber',
            mesh_size=self.mesh_size,
        )
        section.section_calculator.integrator.reference_planes = (
            self.state.reference_planes
        )
        return section

    def solve(
        self,
        n: float = 0.0,
        my: float = 0.0,
        mz: float = 0.0,
        **kwargs,
    ) -> StagedAnalysisResult:
        """Solve a single load case on the current active subset.

        This is strictly read-only on the state: the solved plane is returned
        in the result object and never written back. Calling it repeatedly with
        different loads (e.g. envelopes) does not accumulate any state.

        Arguments:
            n (float): Axial load.
            my (float): Bending moment around y.
            mz (float): Bending moment around z.
            kwargs (dict): Forwarded to ``calculate_strain_profile`` (e.g.
                ``max_iter``, ``tol``, ``initial``).

        Returns:
            StagedAnalysisResult: The result of the solve.
        """
        section = self._build_active_section()
        strain_result = section.section_calculator.calculate_strain_profile(
            n, my, mz, **kwargs
        )
        return StagedAnalysisResult(
            eps_a=strain_result.eps_a,
            chi_y=strain_result.chi_y,
            chi_z=strain_result.chi_z,
            n=n,
            my=my,
            mz=mz,
            section=section,
            reference_planes=self.state.reference_planes,
            active_geometries=self.state.active_geometries,
        )

    def run(self) -> t.List[StagedAnalysisResult]:
        """Run the whole construction sequence.

        Starts from a clean state, then for each stage activates its parts,
        freezes the reference plane of newly activated parts with the strain
        plane solved in the previous stage, and solves the (cumulative) loads.

        After ``run`` the state is left at the final configuration (active set
        and reference planes), ready for read-only envelope solves with
        ``solve``.

        Returns:
            List(StagedAnalysisResult): One result per stage, in order.
        """
        self.state.deactivate_all()
        self.state.clear_all_references()

        results: t.List[StagedAnalysisResult] = []
        last_plane: t.Optional[StrainPlane] = None
        for selector, loads in self._stages:
            geometries = self._resolve_geometries(selector)
            self.freeze_new_active(geometries, last_plane)
            result = self.solve(*loads)
            last_plane = result.strain_plane
            results.append(result)
        return results
