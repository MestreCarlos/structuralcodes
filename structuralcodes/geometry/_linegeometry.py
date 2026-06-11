"""Line geometry implementation as a lightweight wrapper around surface."""

from __future__ import annotations

import typing as t

import numpy as np
from numpy.typing import ArrayLike
from shapely import affinity
from shapely.geometry import LineString

from structuralcodes.core.base import Material
from structuralcodes.materials.basic import ElasticMaterial

from ._geometry import SurfaceGeometry, _mirror_about_axis_matrix


class LineGeometry(SurfaceGeometry):
    """Wrapper geometry defined by centerline + thickness + material."""

    def __init__(
        self,
        line: t.Union[LineString, ArrayLike],
        thickness: float,
        material: Material,
        concrete: bool = False,
        name: t.Optional[str] = None,
        group_label: t.Optional[str] = None,
    ) -> None:
        if thickness <= 0:
            raise ValueError('thickness should be larger than 0')

        if isinstance(line, LineString):
            if len(line.coords) != 2:
                raise ValueError('line should have exactly two points')
            line_obj = line
        else:
            coords = np.asarray(line, dtype=float)
            if coords.shape != (2, 2):
                raise ValueError('line should be provided as shape (2, 2)')
            line_obj = LineString(coords.tolist())

        if line_obj.length <= 0:
            raise ValueError('line length should be larger than 0')

        # Wrapper behavior: convert to equivalent strip polygon.
        poly = line_obj.buffer(
            thickness / 2.0,
            cap_style='flat',
            join_style='mitre',
        )
        super().__init__(
            poly=poly,
            material=material,
            concrete=concrete,
            name=name,
            group_label=group_label,
        )
        self._line = line_obj
        self._thickness = thickness

    @property
    def line(self) -> LineString:
        """Returns the centerline."""
        return self._line

    @property
    def thickness(self) -> float:
        """Returns the thickness."""
        return self._thickness

    @property
    def length(self) -> float:
        """Returns the centerline length."""
        return self._line.length

    def translate(self, dx: float = 0.0, dy: float = 0.0) -> LineGeometry:
        """Returns a new LineGeometry translated by dx, dy."""
        return LineGeometry(
            line=affinity.translate(self.line, dx, dy),
            thickness=self.thickness,
            material=self.material,
            concrete=self.concrete,
            name=self.name,
            group_label=self.group_label,
        )

    def rotate(
        self,
        angle: float = 0.0,
        point: t.Tuple[float, float] = (0.0, 0.0),
        use_radians: bool = True,
    ) -> LineGeometry:
        """Returns a new LineGeometry rotated by angle."""
        return LineGeometry(
            line=affinity.rotate(
                self.line, angle, origin=point, use_radians=use_radians
            ),
            thickness=self.thickness,
            material=self.material,
            concrete=self.concrete,
            name=self.name,
            group_label=self.group_label,
        )

    def mirror(self, axis: LineString) -> LineGeometry:
        """Returns a new LineGeometry mirrored about the given axis."""
        if not isinstance(axis, LineString):
            raise TypeError('axis should be a shapely LineString object')
        A = _mirror_about_axis_matrix(axis)
        params = [A[0, 0], A[0, 1], A[1, 0], A[1, 1], A[0, 2], A[1, 2]]
        return LineGeometry(
            line=affinity.affine_transform(self.line, params),
            thickness=self.thickness,
            material=self.material,
            concrete=self.concrete,
            name=self.name,
            group_label=self.group_label,
        )

    @staticmethod
    def from_geometry(
        geo: LineGeometry,
        new_material: t.Optional[Material] = None,
    ) -> LineGeometry:
        """Create a new LineGeometry with a different material."""
        if not isinstance(geo, LineGeometry):
            raise TypeError('geo should be a LineGeometry')
        if new_material is not None:
            if not isinstance(new_material, Material):
                raise TypeError(
                    f'new_material should be a valid structuralcodes.base.\
                    Material object. \
                    {repr(new_material)}'
                )
        else:
            new_material = ElasticMaterial.from_material(geo.material)

        return LineGeometry(
            line=geo.line,
            thickness=geo.thickness,
            material=new_material,
            concrete=geo.concrete,
            name=geo.name,
            group_label=geo.group_label,
        )
