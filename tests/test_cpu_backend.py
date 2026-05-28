from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from wifi_optimizer.fallback_backend import count_segment_intersections
from wifi_optimizer.mesh_utils import prepare_mesh
from wifi_optimizer.optimizer import generate_candidate_grid, optimize_placement
from wifi_optimizer.rf_math import fspl_db


class CpuBackendTests(unittest.TestCase):
    def test_fspl_is_finite(self) -> None:
        self.assertGreater(fspl_db(1.0, 5.8e9), 40.0)

    def test_segment_intersects_wall_once(self) -> None:
        vertices = np.array(
            [
                [0.0, -1.0, -1.0],
                [0.0, 1.0, -1.0],
                [0.0, 1.0, 1.0],
                [0.0, -1.0, 1.0],
            ],
            dtype=float,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        intersections = count_segment_intersections(np.asarray(mesh.triangles), np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        self.assertEqual(intersections, 1)

    def test_fallback_optimizer_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mesh_path = tmp_path / "room.ply"
            mesh = trimesh.creation.box(extents=(3.0, 3.0, 2.5))
            mesh.export(mesh_path)
            prepared, stats = prepare_mesh(mesh_path, tmp_path / "out", max_faces=0)
            candidates = generate_candidate_grid(
                np.asarray(stats.bounds_min),
                np.asarray(stats.bounds_max),
                grid_size=(2, 2),
                z_m=0.8,
                margin_m=0.2,
            )
            receivers = [{"name": "Rx1", "position_m": [0.0, 0.0, 0.2]}]
            result = optimize_placement(prepared, candidates, receivers, 5.8e9, 20.0, backend="fallback")
            self.assertEqual(result.path_loss_db.shape, (4, 1))
            self.assertTrue(np.isfinite(result.worst_case_path_loss_db).all())


if __name__ == "__main__":
    unittest.main()
