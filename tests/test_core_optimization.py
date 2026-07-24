from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import core  # noqa: E402


def two_triangle_fixture(seam: bool = False):
    loop_vert = np.array([0, 1, 2, 2, 1, 3], np.int32)
    loop_start = np.array([0, 3], np.int32)
    loop_total = np.array([3, 3], np.int32)
    loop_uv = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5] if seam else [0.0, 1.0],
            [0.9, 0.2] if seam else [1.0, 0.0],
            [1.0, 1.0],
        ],
        np.float32,
    )
    gx = np.full((16, 16), 0.7, np.float32)
    gy = np.full((16, 16), -0.2, np.float32)
    weight = np.ones((16, 16), np.float32)
    loop_edge = np.array([0, 1, 2, 1, 3, 4], np.int32)
    return (
        loop_vert,
        loop_uv,
        loop_start,
        loop_total,
        gx,
        gy,
        weight,
        loop_edge,
    )


class GradientConstraintCompressionTests(unittest.TestCase):
    def constraints(self, seam: bool, merge_duplicates: bool):
        (
            loop_vert,
            loop_uv,
            loop_start,
            loop_total,
            gx,
            gy,
            weight,
            loop_edge,
        ) = two_triangle_fixture(seam)
        return core.gradient_constraints_from_loops(
            loop_vert,
            loop_uv,
            loop_start,
            loop_total,
            gx,
            gy,
            weight,
            merge_duplicates=merge_duplicates,
            loop_edge=loop_edge,
        )

    def test_shared_non_seam_edge_is_merged(self):
        raw = self.constraints(seam=False, merge_duplicates=False)
        merged = self.constraints(seam=False, merge_duplicates=True)
        self.assertEqual(len(raw[0]), 6)
        self.assertEqual(len(merged[0]), 5)

    def test_uv_seam_keeps_both_observations(self):
        merged = self.constraints(seam=True, merge_duplicates=True)
        self.assertEqual(len(merged[0]), 6)

    def test_compression_preserves_joint_least_squares_solution(self):
        raw = self.constraints(seam=False, merge_duplicates=False)
        merged = self.constraints(seam=False, merge_duplicates=True)
        raw_solution, _ = core.solve_joint_position_normal(
            *raw[:3],
            vert_count=4,
            base_weight=raw[3],
            position_weight=0.1,
            irls_iters=3,
            max_iter=2000,
            tolerance=1e-10,
        )
        merged_solution, _ = core.solve_joint_position_normal(
            *merged[:3],
            vert_count=4,
            base_weight=merged[3],
            position_weight=0.1,
            irls_iters=3,
            max_iter=2000,
            tolerance=1e-10,
        )
        np.testing.assert_allclose(
            merged_solution,
            raw_solution,
            rtol=1e-6,
            atol=1e-7,
        )


class ContentDigestTests(unittest.TestCase):
    def test_digest_is_stable_for_equal_array_copies(self):
        original = np.arange(24, dtype=np.float32).reshape(3, 8)
        self.assertEqual(
            core.content_digest(original),
            core.content_digest(original.copy()),
        )

    def test_digest_changes_with_geometry_uv_or_gradient_content(self):
        geometry = np.arange(12, dtype=np.float32).reshape(4, 3)
        uv = np.arange(8, dtype=np.float32).reshape(4, 2)
        gradient = np.ones((4, 4), np.float32)
        baseline = core.content_digest(geometry, uv, gradient)
        for index, source in enumerate((geometry, uv, gradient)):
            arrays = [geometry.copy(), uv.copy(), gradient.copy()]
            arrays[index].flat[-1] += np.float32(0.25)
            self.assertNotEqual(baseline, core.content_digest(*arrays))

    def test_digest_includes_shape_and_dtype(self):
        values = np.arange(8, dtype=np.float32)
        self.assertNotEqual(
            core.content_digest(values),
            core.content_digest(values.reshape(2, 4)),
        )
        self.assertNotEqual(
            core.content_digest(values),
            core.content_digest(values.astype(np.float64)),
        )


class JointSolverControlTests(unittest.TestCase):
    def test_cancellation_is_checked_before_iterations(self):
        with self.assertRaises(core.JointSolveCancelled):
            core.solve_joint_position_normal(
                np.array([0], np.int32),
                np.array([1], np.int32),
                np.array([1.0], np.float32),
                vert_count=2,
                cancel_check=lambda: True,
            )

    def test_progress_reports_completion_budget(self):
        progress = []
        core.solve_joint_position_normal(
            np.array([0], np.int32),
            np.array([1], np.int32),
            np.array([1.0], np.float32),
            vert_count=2,
            position_weight=0.1,
            irls_iters=1,
            max_iter=20,
            tolerance=1e-8,
            progress_callback=lambda done, total: progress.append((done, total)),
        )
        self.assertTrue(progress)
        self.assertEqual(progress[-1], (40, 40))


if __name__ == "__main__":
    unittest.main()
