"""Compare LineGeometry and SurfaceGeometry for bending and performance.

This script does three things:
1) Creates an I-like section (one flange + two webs) with LineGeometry.
2) Creates the same section with SurfaceGeometry.
3) Benchmarks a larger case with 300 LineGeometries vs 300 SurfaceGeometries.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

from shapely.geometry import Polygon

# Allow running the example directly from the repository root.
if __package__ in (None, ''):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

from structuralcodes.geometry import (
    CompoundGeometry,
    LineGeometry,
    SurfaceGeometry,
)
from structuralcodes.materials.basic import ElasticPlasticMaterial
from structuralcodes.sections import GenericSection


def create_i_section_line(
    material: ElasticPlasticMaterial,
) -> CompoundGeometry:
    """Create an I-like section with one flange and two webs (line model)."""
    flange = LineGeometry(
        line=((-150.0, 150.0), (150.0, 150.0)),
        thickness=20.0,
        material=material,
    )
    web_left = LineGeometry(
        line=((-75.0, -150.0), (-75.0, 150.0)),
        thickness=12.0,
        material=material,
    )
    web_right = LineGeometry(
        line=((75.0, -150.0), (75.0, 150.0)),
        thickness=12.0,
        material=material,
    )
    return CompoundGeometry([flange, web_left, web_right])


def create_i_section_surface(
    material: ElasticPlasticMaterial,
) -> CompoundGeometry:
    """Create the same I-like section using SurfaceGeometry rectangles."""
    flange = SurfaceGeometry(
        poly=Polygon(
            (
                (-150.0, 140.0),
                (150.0, 140.0),
                (150.0, 160.0),
                (-150.0, 160.0),
            )
        ),
        material=material,
    )
    web_left = SurfaceGeometry(
        poly=Polygon(
            (
                (-81.0, -150.0),
                (-69.0, -150.0),
                (-69.0, 150.0),
                (-81.0, 150.0),
            )
        ),
        material=material,
    )
    web_right = SurfaceGeometry(
        poly=Polygon(
            (
                (69.0, -150.0),
                (81.0, -150.0),
                (81.0, 150.0),
                (69.0, 150.0),
            )
        ),
        material=material,
    )
    return CompoundGeometry([flange, web_left, web_right])


def calculate_bending_resistance(
    geometry: CompoundGeometry, integrator: str
) -> float:
    """Compute My bending resistance (theta=0, n=0)."""
    section = GenericSection(
        geometry=geometry, integrator=integrator, mesh_size=0.01
    )
    result = section.section_calculator.calculate_bending_strength(
        theta=0, n=0
    )
    return float(result.m_y)


def create_many_lines(
    material: ElasticPlasticMaterial, n_rows: int = 15, n_cols: int = 20
) -> CompoundGeometry:
    """Create a section with n_rows*n_cols LineGeometries."""
    geoms = []
    length = 200.0
    thickness = 8.0
    pitch_x = 22.0
    pitch_y = 22.0
    x0 = -0.5 * (n_cols - 1) * pitch_x
    y0 = -0.5 * (n_rows - 1) * pitch_y
    for j in range(n_rows):
        for i in range(n_cols):
            xc = x0 + i * pitch_x
            yc = y0 + j * pitch_y
            geoms.append(
                LineGeometry(
                    line=((xc - 0.5 * length, yc), (xc + 0.5 * length, yc)),
                    thickness=thickness,
                    material=material,
                )
            )
    return CompoundGeometry(geoms)


def create_many_surfaces(
    material: ElasticPlasticMaterial, n_rows: int = 15, n_cols: int = 20
) -> CompoundGeometry:
    """Create an equivalent section with n_rows*n_cols SurfaceGeometries."""
    geoms = []
    length = 200.0
    thickness = 8.0
    pitch_x = 22.0
    pitch_y = 22.0
    x0 = -0.5 * (n_cols - 1) * pitch_x
    y0 = -0.5 * (n_rows - 1) * pitch_y
    for j in range(n_rows):
        for i in range(n_cols):
            xc = x0 + i * pitch_x
            yc = y0 + j * pitch_y
            geoms.append(
                SurfaceGeometry(
                    poly=Polygon(
                        (
                            (xc - 0.5 * length, yc - 0.5 * thickness),
                            (xc + 0.5 * length, yc - 0.5 * thickness),
                            (xc + 0.5 * length, yc + 0.5 * thickness),
                            (xc - 0.5 * length, yc + 0.5 * thickness),
                        )
                    ),
                    material=material,
                )
            )
    return CompoundGeometry(geoms)


def benchmark_bending(
    geometry: CompoundGeometry, integrator: str, repeats: int = 3
) -> float:
    """Return average time [s] to compute bending resistance."""
    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        _ = calculate_bending_resistance(geometry, integrator)
        times.append(perf_counter() - t0)
    return sum(times) / len(times)


if __name__ == '__main__':
    # Steel with elastic-plastic law and ultimate strain for strength analysis.
    steel = ElasticPlasticMaterial(
        E=210000.0, fy=355.0, density=7850.0, eps_su=0.15
    )

    integrator = 'marin'

    print('--- I-like section: LineGeometry vs SurfaceGeometry ---')
    i_line = create_i_section_line(steel)
    i_surface = create_i_section_surface(steel)

    my_line = calculate_bending_resistance(i_line, integrator=integrator)
    my_surface = calculate_bending_resistance(i_surface, integrator=integrator)
    rel_diff = (my_line - my_surface) / my_surface * 100.0

    print(f'My (LineGeometry):    {my_line:,.3f} Nmm')
    print(f'My (SurfaceGeometry): {my_surface:,.3f} Nmm')
    print(f'Relative difference:  {rel_diff:.3f} %')

    large_lines = create_many_lines(steel, n_rows=30, n_cols=30)
    large_surfaces = create_many_surfaces(steel, n_rows=30, n_cols=30)
    print('\n--- Performance benchmark (300 geometries) ---')
    t_line = benchmark_bending(large_lines, repeats=3, integrator=integrator)
    t_surface = benchmark_bending(
        large_surfaces, repeats=3, integrator=integrator
    )
    speedup = t_surface / t_line

    print(f'Integrator: {integrator}')
    print(f'Average time LineGeometry:    {t_line:.4f} s')
    print(f'Average time SurfaceGeometry: {t_surface:.4f} s')
    print(f'Speed-up (Surface/Line):      {speedup:.2f}x')
