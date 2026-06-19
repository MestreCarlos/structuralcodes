"""Tests for the staged (evolutive) sectional analysis."""

import numpy as np
import pytest
from shapely import Polygon

from structuralcodes.geometry import CompoundGeometry, SurfaceGeometry
from structuralcodes.materials.basic import ElasticMaterial
from structuralcodes.sections import (
    GenericSection,
    SectionStrainState,
    StagedSectionCalculator,
)


def _composite_geometry():
    """Return a two-part composite geometry (steel + slab) and the parts.

    The steel sits in z in [0, 200] and the slab on top in z in [200, 400].
    """
    steel_mat = ElasticMaterial(E=210000, density=7850)
    concrete_mat = ElasticMaterial(E=30000, density=2500)
    steel = SurfaceGeometry(
        Polygon(((-100, 0), (100, 0), (100, 200), (-100, 200))),
        steel_mat,
        group_label='steel',
    )
    slab = SurfaceGeometry(
        Polygon(((-150, 200), (150, 200), (150, 400), (-150, 400))),
        concrete_mat,
        group_label='slab',
    )
    compound = CompoundGeometry([steel, slab])
    return compound, steel, slab


def _slab_point():
    """Return a representative (y, z) point inside the slab."""
    return 0.0, 300.0


# -- State container -----------------------------------------------------
def test_strain_state_lifecycle():
    """SectionStrainState basic lifecycle operations."""
    compound, steel, slab = _composite_geometry()
    state = SectionStrainState(
        list(compound.geometries) + list(compound.point_geometries)
    )

    # Default: everything active, no predeformation.
    assert state.is_active(steel)
    assert state.is_active(slab)
    assert not state.has_predeformation
    assert set(state.active_geometries) == {steel, slab}

    # Deactivate slab.
    state.deactivate(slab)
    assert not state.is_active(slab)
    assert state.active_geometries == [steel]

    # Set a reference plane for the slab.
    state.set_reference_plane(slab, (1e-4, 2e-7, 0.0))
    assert state.has_predeformation
    assert state.reference_plane(slab) == (1e-4, 2e-7, 0.0)
    assert state.reference_plane(steel) is None

    # Snapshot and restore.
    snap = state.snapshot()
    state.reset()
    assert not state.has_predeformation
    assert state.is_active(slab)
    state.restore(snap)
    assert state.has_predeformation
    assert not state.is_active(slab)


# -- Regression: no state == plain GenericSection -------------------------
def test_solve_without_state_matches_generic_section():
    """With no staging, solve must match a plain fiber GenericSection."""
    compound, _, _ = _composite_geometry()

    staged = StagedSectionCalculator(compound, mesh_size=0.01)
    assert not staged.has_predeformation
    result = staged.solve(n=0.0, my=5e6, mz=0.0)

    section = GenericSection(compound, integrator='fiber', mesh_size=0.01)
    expected = section.section_calculator.calculate_strain_profile(
        0.0, 5e6, 0.0
    )

    assert np.allclose(
        result.strain_plane, expected.strain_plane, rtol=1e-9, atol=1e-12
    )


# -- Envelopes do not accumulate state -----------------------------------
def test_envelope_solves_are_independent():
    """Repeated read-only solves do not accumulate or contaminate state."""
    compound, _, _ = _composite_geometry()
    staged = StagedSectionCalculator(compound, mesh_size=0.01)

    res_a = staged.solve(my=5e6)
    res_b = staged.solve(my=-5e6)
    res_a_again = staged.solve(my=5e6)

    # Same load -> same result regardless of order / previous solves.
    assert np.allclose(res_a.strain_plane, res_a_again.strain_plane)
    # Different load -> different result.
    assert not np.allclose(res_a.strain_plane, res_b.strain_plane)
    # solve never writes any state.
    assert not staged.has_predeformation


def test_solve_without_active_geometries_raises():
    """Solving with no active geometry raises a clear error."""
    compound, _, _ = _composite_geometry()
    staged = StagedSectionCalculator(compound, mesh_size=0.01)
    staged.state.deactivate_all()
    with pytest.raises(ValueError):
        staged.solve(my=1e6)


# -- Staged analysis: slab added without extra load -----------------------
def test_staged_slab_added_without_extra_load_carries_no_stress():
    """Activating the slab without extra load leaves it stress-free.

    The strain plane must not change (the slab does not stiffen the response
    to the already applied loads) and the slab driving strain must be ~0.
    """
    compound, _, slab = _composite_geometry()
    staged = StagedSectionCalculator(compound, mesh_size=0.01)

    m1 = 5e6
    staged.add_stage('steel', my=m1)
    staged.add_stage(['steel', 'slab'], my=m1)  # same cumulative load
    results = staged.run()

    plane_1 = np.array(results[0].strain_plane)
    plane_2 = np.array(results[1].strain_plane)

    # The slab got a reference plane frozen at phase 1.
    assert staged.has_predeformation
    assert staged.state.reference_plane(slab) is not None

    # Plane is essentially unchanged.
    assert np.allclose(plane_1, plane_2, rtol=1e-3, atol=1e-10)

    # The slab feels no driving strain (P2 - P1 ~ 0).
    y, z = _slab_point()
    eps_slab = results[1].driving_strain(y, z, geometry=slab)
    assert abs(eps_slab) < 1e-7


# -- Staged analysis: slab takes the load increment -----------------------
def test_staged_slab_takes_load_increment():
    """When extra load is applied after activating the slab, it takes it."""
    compound, steel, slab = _composite_geometry()
    staged = StagedSectionCalculator(compound, mesh_size=0.01)

    m1 = 5e6
    m2 = 2.0e7  # larger cumulative load at phase 2
    staged.add_stage('steel', my=m1)
    staged.add_stage(['steel', 'slab'], my=m2)
    results = staged.run()

    # Phase 2 equilibrium: internal resultants match the applied loads.
    n_int, my_int, mz_int = results[1].stress_resultants()
    assert abs(n_int) < 1e-3 * m2 / 100 + 1.0
    assert np.isclose(my_int, m2, rtol=1e-2)
    assert abs(mz_int) < 1e-2 * m2

    # The slab now carries a non-negligible driving strain.
    y, z = _slab_point()
    eps_slab = results[1].driving_strain(y, z, geometry=slab)
    assert abs(eps_slab) > 1e-6

    # The steel feels the full plane (no reference subtracted).
    eps_steel = results[1].driving_strain(0.0, 100.0, geometry=steel)
    plane = results[1].strain_plane
    assert np.isclose(eps_steel, plane[0] + plane[1] * 100.0)


# -- Reset returns to the virgin model ------------------------------------
def test_reset_after_run():
    """reset() clears references and reactivates all parts."""
    compound, _, _ = _composite_geometry()
    staged = StagedSectionCalculator(compound, mesh_size=0.01)
    staged.add_stage('steel', my=5e6)
    staged.add_stage(['steel', 'slab'], my=1e7)
    staged.run()

    assert staged.has_predeformation
    staged.reset()
    assert not staged.has_predeformation
    assert len(staged.state.active_geometries) == 2
