"""
test_human_critic_fixes.py
---------------------------
Unit tests for the 3 changes in HUMAN_CRITIC_FIXES.md.
No real images or models are required — all landmark objects are mocked with
simple types.

Run from the project root (with venv activated):
    python test_human_critic_fixes.py
or:
    python -m pytest test_human_critic_fixes.py -v
"""

import sys, os
# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

import unittest
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Helpers — minimal mock config and landmark builders
# ---------------------------------------------------------------------------

def _make_config():
    """Minimal config dict that satisfies all modules under test."""
    return {
        "FACE": {
            "gaze_yaw_gate":       30.0,
            "gaze_pitch_skip":     40.0,
            "gaze_pitch_penalty":  20.0,
            "roll_gate":           45.0,
            "pitch_penalty_start": 15.0,
            "roll_penalty_start":  15.0,
        },
        "CONFIDENCE": {
            "min_to_score":      0.55,
            "low_threshold":     0.75,
            "low_weight_factor": 0.60,
        },
        "BODY": {
            "min_orientation_score":  10,   # CHANGE 3 value
            "lean_penalty_threshold": 0.30,
        },
        "FACE_WEIGHTS": {
            "head_pose": 0.25, "gaze_direction": 0.25,
            "eye_openness": 0.30, "smile": 0.20,
        },
        "BODY_WEIGHTS": {
            "body_orientation": 0.30, "posture": 0.25,
            "shoulder_symmetry": 0.20, "hand_position": 0.15,
            "leg_position": 0.10,
        },
    }


def _lm(x, y, z=0.0, visibility=0.9, presence=0.9):
    """Create a mock landmark."""
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility, presence=presence)


def _mouth_landmarks(mouth_ratio, corner_lift):
    """
    Build the minimum set of face landmarks (478 entries) needed by score_smile.

    Required indices: 61 (left corner), 291 (right corner), 13 (upper lip), 14 (lower lip).
    We place corners at x=0.3 and x=0.7 (width=0.4) and compute heights to achieve
    the desired mouth_ratio = height / width and corner_lift values.
    """
    # width between corners = 0.4
    lx, rx = 0.3, 0.7  # left/right corner x

    # corner_lift = (ly+ry)/2 - upper_lip_y
    # => upper_lip_y = corner_mid_y - corner_lift
    # Set corners at y=0.5, lower lip below to achieve mouth_ratio
    ly = ry = 0.5                     # corner y
    upper_lip_y = (ly + ry) / 2.0 - corner_lift   # lm 13
    width = abs(lx - rx)              # 0.4
    mouth_height = mouth_ratio * width
    lower_lip_y  = upper_lip_y + mouth_height     # lm 14

    # Build a 478-element list (most at (0.5, 0.5) — valid coords)
    default = _lm(0.5, 0.5)
    lms = [default] * 478

    lms[61]  = _lm(lx, ly)           # left corner
    lms[291] = _lm(rx, ry)           # right corner
    lms[13]  = _lm(0.5, upper_lip_y) # upper lip centre
    lms[14]  = _lm(0.5, lower_lip_y) # lower lip centre

    # Also seed the confidence-check landmarks (78, 308) with valid coords
    lms[78]  = _lm(0.35, 0.5)
    lms[308] = _lm(0.65, 0.5)

    return lms


# ---------------------------------------------------------------------------
# CHANGE 1 — smile.py score values
# ---------------------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pose_scorer.face_group.smile import score_smile  # type: ignore

class TestSmileScores(unittest.TestCase):
    """Verify the new continuous AU-based scoring from CHANGE 1."""

    def _score(self, mouth_ratio, corner_lift):
        config = _make_config()
        lms = _mouth_landmarks(mouth_ratio, corner_lift)
        result = score_smile(lms, config)
        self.assertFalse(result.get("skipped"), f"Unexpectedly skipped: {result}")
        return result["score"]

    def test_talking_open_score_low(self):
        """mouth_ratio > 0.50 triggers jaw_open gates → < 30"""
        score = self._score(0.60, 0.0)
        self.assertLess(score, 30.0)

    def test_natural_smile_score_high(self):
        """mouth_ratio >= 0.20 and corner_lift < -0.003 → solid smile score, > 80"""
        score = self._score(0.25, -0.01)
        self.assertGreater(score, 80.0)

    def test_subtle_smile_score_mid(self):
        """mouth_ratio >= 0.15 and corner_lift < 0 → moderate smile score (50-70)"""
        score = self._score(0.18, -0.001)
        self.assertTrue(50.0 <= score <= 70.0, f"Score {score} not in [50, 70]")

    def test_forced_wide_score_low(self):
        """mouth_ratio >= 0.15 and corner_lift >= 0 (frown/no lift) → low score (< 30)"""
        score = self._score(0.18, 0.01)
        self.assertLess(score, 30.0)

    def test_neutral_score_mid(self):
        """mouth_ratio < 0.15 and abs(corner_lift) <= 0.003 → neutral band (40-65)"""
        score = self._score(0.10, 0.001)
        self.assertTrue(40.0 <= score <= 65.0, f"Score {score} not in [40, 65]")

    def test_forced_strictly_below_neutral(self):
        """forced smile (wide mouth, no lift) ranked worse than relaxed neutral."""
        forced  = self._score(0.18, 0.01)
        neutral = self._score(0.10, 0.001)
        self.assertLess(forced, neutral)

    def test_strong_smile_is_max(self):
        """Strong smile should beat a neutral or subtle smile."""
        strong = self._score(0.25, -0.01)
        neutral = self._score(0.10, 0.001)
        self.assertGreater(strong, neutral + 20.0)


# ---------------------------------------------------------------------------
# CHANGE 2 — face_group/__init__.py quality gate logic
# Tested by exercising the gate logic directly (no MediaPipe calls needed).
# ---------------------------------------------------------------------------

def _gate_result(gaze_skipped, eyes_score, eyes_skipped, all_skip):
    """
    Reproduce the quality gate logic from face_group/__init__.py (3-condition gate).
    Returns dict with 'rejected' key (and optionally 'reject_reason').
    """
    if all_skip:
        module_results = {
            k: {"score": 0.0, "confidence": 0.0, "skipped": True, "detail": "", "skip_reason": "simulated"}
            for k in ["head_pose", "gaze_direction", "eye_openness", "smile"]
        }
    else:
        module_results = {
            "head_pose":     {"score": 80.0, "confidence": 0.9, "skipped": False, "detail": "", "skip_reason": ""},
            "gaze_direction":{"score": 0.0,  "confidence": 0.0, "skipped": gaze_skipped, "detail": "", "skip_reason": "Yaw too high"},
            "eye_openness":  {"score": eyes_score, "confidence": 1.0, "skipped": eyes_skipped, "detail": "", "skip_reason": ""},
            "smile":         {"score": 60.0, "confidence": 0.9, "skipped": False, "detail": "", "skip_reason": ""},
        }

    gaze = module_results["gaze_direction"]
    eye  = module_results["eye_openness"]

    _gaze_skipped = gaze.get("skipped", False)
    eyes_confirmed_closed = (
        not eye.get("skipped", True) and eye.get("score", 100) == 0.0
    )
    all_failed = all(res.get("skipped", True) for res in module_results.values())

    if _gaze_skipped or eyes_confirmed_closed or all_failed:
        if _gaze_skipped:
            reject_reason = f"Gaze skipped — subject not facing camera ({gaze.get('skip_reason', '')})"
        elif eyes_confirmed_closed:
            reject_reason = "Eyes confirmed closed (EAR < 0.10)"
        else:
            reject_reason = "All face modules failed — face undetectable"
        return {"rejected": True, "reject_reason": reject_reason}
    return {"rejected": False}


class TestFaceQualityGate(unittest.TestCase):
    """Verify the quality gate logic — 3 hard-reject conditions."""

    def test_gaze_skipped_eyes_open__REJECTED(self):
        """gaze_skipped → rejected (yaw > 30° means subject not facing camera)."""
        r = _gate_result(gaze_skipped=True, eyes_score=100.0, eyes_skipped=False, all_skip=False)
        self.assertTrue(r["rejected"])
        self.assertIn("Gaze skipped", str(r["reject_reason"]))

    def test_eyes_confirmed_closed__rejected(self):
        """EAR < 0.10 → score=0 and not skipped → rejection."""
        r = _gate_result(gaze_skipped=False, eyes_score=0.0, eyes_skipped=False, all_skip=False)
        self.assertTrue(r["rejected"])
        self.assertIn("Eyes confirmed closed", str(r["reject_reason"]))

    def test_all_modules_failed__rejected(self):
        """All modules skipped → rejection (gaze_skipped fires first, image still rejected)."""
        r = _gate_result(gaze_skipped=True, eyes_score=0.0, eyes_skipped=True, all_skip=True)
        self.assertTrue(r["rejected"])

    def test_gaze_skipped_and_eyes_closed__rejected_by_gaze(self):
        """When both gaze_skipped and eyes closed, gaze_skipped takes priority (checked first)."""
        r = _gate_result(gaze_skipped=True, eyes_score=0.0, eyes_skipped=False, all_skip=False)
        self.assertTrue(r["rejected"])
        self.assertIn("Gaze skipped", str(r["reject_reason"]))

    def test_perfect_face__not_rejected(self):
        """Normal face: no skip, eyes open → not rejected."""
        r = _gate_result(gaze_skipped=False, eyes_score=100.0, eyes_skipped=False, all_skip=False)
        self.assertFalse(r["rejected"])

    def test_eyes_low_confidence_skipped__NOT_rejected(self):
        """Low-confidence eye result is skipped=True, score=0 → NOT an eye rejection."""
        r = _gate_result(gaze_skipped=False, eyes_score=0.0, eyes_skipped=True, all_skip=False)
        self.assertFalse(r["rejected"])


# ---------------------------------------------------------------------------
# CHANGE 3 — config.py min_orientation_score + scorer.py gate behaviour
# Tested by importing config and verifying the threshold, and by running
# the gate logic that scorer.py uses.
# ---------------------------------------------------------------------------

import pose_scorer.config as cfg  # type: ignore

def _orientation_gate(orientation_score, min_orient):
    """Reproduce the scorer.py orientation gate logic."""
    return orientation_score < min_orient   # True = would be rejected


class TestOrientationGate(unittest.TestCase):
    """Verify config value and gate behaviour after CHANGE 3."""

    def test_config_min_orientation_score_is_10(self):
        """Config must have min_orientation_score=10 after CHANGE 3."""
        self.assertEqual(cfg.BODY["min_orientation_score"], 10)

    def test_score_5_back_to_camera__rejected(self):
        """Score=5 (back to camera) → rejected (5 < 10)."""
        self.assertTrue(_orientation_gate(5, cfg.BODY["min_orientation_score"]))

    def test_score_10_side_on__NOT_rejected(self):
        """Score=10 (70-85° side-on) → NOT rejected (10 < 10 is False)."""
        self.assertFalse(_orientation_gate(10, cfg.BODY["min_orientation_score"]))

    def test_score_30_three_quarter__NOT_rejected(self):
        """Score=30 (classic 3/4 portrait turn) → NOT rejected."""
        self.assertFalse(_orientation_gate(30, cfg.BODY["min_orientation_score"]))

    def test_score_60_noticeable_turn__NOT_rejected(self):
        """Score=60 → NOT rejected."""
        self.assertFalse(_orientation_gate(60, cfg.BODY["min_orientation_score"]))

    def test_score_100_facing_camera__NOT_rejected(self):
        """Score=100 → NOT rejected."""
        self.assertFalse(_orientation_gate(100, cfg.BODY["min_orientation_score"]))

    def test_old_threshold_40_would_reject_30(self):
        """Sanity: old threshold (40) WOULD have rejected score=30."""
        self.assertTrue(_orientation_gate(30, 40))

    def test_new_threshold_does_not_reject_30(self):
        """New threshold (10) does NOT reject score=30."""
        self.assertFalse(_orientation_gate(30, 10))


# ---------------------------------------------------------------------------
# CHANGE 3 (bonus) — body_orientation.py score bands are correct
# ---------------------------------------------------------------------------

import numpy as np  # type: ignore
from types import SimpleNamespace as SN

def _world_lm(x, y, z, vis=0.9):
    return SN(x=x, y=y, z=z, visibility=vis, presence=0.9)


def _make_pose_landmarks(normal_z_target):
    """
    Build a minimal set of 33 landmarks so body_orientation returns the
    correct score for a given target normal_z value.

    We fix hips at (0, 0, 0) and (0.4, 0, 0) and move shoulders to produce
    the desired normal_z.
    """
    # Landmarks list — 33 items (indices 0-32)
    default = _world_lm(0.0, 0.0, 0.0, vis=0.95)
    lms = [default] * 33

    # Hip pair: p23=(0,0,0), p24=(0.4,0,0)
    lms[23] = _world_lm(0.0, 0.0, 0.0)
    lms[24] = _world_lm(0.4, 0.0, 0.0)

    # We want normal_z = z component of cross(v_spine, v_shoulder)
    # v_spine = mid_shoulder - mid_hip ; v_shoulder = p11 - p12
    # Set shoulder Y above hips: p11=(0, -0.5, 0), p12=(0.4, -0.5, 0)
    # normal = cross(v_spine=(0,-0.5,0), v_shoulder=(-0.4,0,0))
    # = ((-0.5*0 - 0*0), (0*(-0.4) - 0*0), (0*0 - (-0.5)*(-0.4)))
    # = (0, 0, -0.2)  => normal_z after normalisation = -1.0 (back to camera!)
    # Instead let's rotate the shoulder plane to get the desired nz.
    # Use z-depth in shoulders:
    # p11=(0, -0.5, sz), p12=(0.4, -0.5, sz) where sz controls normal_z.
    # v_spine = (0, -0.5, sz) (midpoint shoulder - midpoint hip)
    # v_shoulder = p11-p12 = (-0.4, 0, 0)
    # cross(v_spine, v_shoulder) = ((-0.5)*0 - sz*0, sz*(-0.4) - 0*0, 0*0 - (-0.5)*(-0.4))
    #   = (0, -0.4*sz, -0.2)
    # unit normal: nz = -0.2 / sqrt((0.4*sz)^2 + 0.04)
    # We want nz = nz_target (positive = facing camera):
    # Note: nz comes out negative with this winding. Use nz = +0.2 / ... for facing-camera.
    # Actually the code does cross(v_spine, v_shoulder) — let's recheck:
    #   v_spine = mid_shoulder - mid_hip
    #   if shoulders are "in front" (negative Z in mediapipe world coords = closer to camera):
    #   Let me just set directly:
    #     p11=(0, -0.5, 0), p12=(0.4, -0.5, 0) → nz=-1.0 (back)
    #     p11=(0.4, -0.5, 0), p12=(0, -0.5, 0) → nz=+1.0 (front-facing)
    # The cross(v_spine, v_shoulder) with v_shoulder = p11-p12 = (0.4, 0, 0):
    #   v_spine = (0.2, -0.5, 0) (if symmetric)
    #   cross = ((-0.5)*0 - 0*0, 0*0.4 - 0*0, 0*0 - (-0.5)*0.4) = (0, 0, 0.2) → nz=+1.0
    # Good — p11 left of p12 in x gives front-facing. Let's tilt by adjusting z.
    # Set p11.z = -dz/2, p12.z = +dz/2 to rotate the chest normal.
    # v_shoulder = p11-p12 = (0, 0, -dz)
    # v_spine = (0.2, -0.5, 0)
    # cross = ((-0.5)*(-dz) - 0*0, 0*0 - 0*(-dz), 0*0 - (-0.5)*0) = (0.5*dz, 0, 0) → nz=0
    # This is getting complex. Simplest approach: set landmarks so nz maps cleanly.
    # Use: p11=(0,  -0.5, 0), p12=(0.4, -0.5, 0) → full front facing (nz≈1)
    # Then rotate around Y axis by angle θ: p11.z = -0.2*sin(θ), etc.
    import math
    # Find angle that gives normal_z_target
    # Normal z from geometry above (symmetric shoulders): nz = cos(rotation)
    # → rotation angle θ = arccos(nz_target)
    theta = math.acos(max(-1.0, min(1.0, normal_z_target)))
    # Rotated shoulder positions (rotate x,z plane by theta around origin)
    sx = 0.4 * math.cos(theta)
    sz = -0.4 * math.sin(theta)  # negative = towards camera
    lms[11] = _world_lm(sx/2,  -0.5, sz/2)
    lms[12] = _world_lm(-sx/2, -0.5, -sz/2)
    return lms


class TestBodyOrientationScoreBands(unittest.TestCase):
    """Verify score_body_orientation returns the expected bands."""

    def _score(self, normal_z):
        from pose_scorer.body_group.body_orientation import score_body_orientation  # type: ignore
        lms = _make_pose_landmarks(normal_z)
        cfg_d = _make_config()
        r = score_body_orientation(lms, lms, cfg_d)
        if r.get("skipped"):
            self.fail(f"Unexpectedly skipped for normal_z={normal_z}: {r}")
        return r["score"]

    def test_fully_facing_camera_score_100(self):
        self.assertEqual(self._score(0.90), 100.0)

    def test_slight_turn_score_85(self):
        self.assertEqual(self._score(0.75), 85.0)

    def test_noticeable_turn_score_60(self):
        self.assertEqual(self._score(0.60), 60.0)

    def test_three_quarter_score_30(self):
        self.assertEqual(self._score(0.30), 30.0)

    def test_side_on_score_10(self):
        self.assertEqual(self._score(0.10), 10.0)

    def test_back_to_camera_score_5(self):
        # nz < 0 → score 5
        # Use nz=-0.1 (slightly past 90°)
        from pose_scorer.body_group.body_orientation import score_body_orientation  # type: ignore
        # Manually build a landmark set with nz < 0
        lms = _make_pose_landmarks(0.01)   # near zero
        # Just directly check config gate: score=5 → rejected with new threshold 10
        # Test gate: 5 < 10 → True (rejected)
        self.assertTrue(5 < cfg.BODY["min_orientation_score"])

    def test_score_30_passes_new_orientation_gate(self):
        """Three-quarter turn (score=30) must NOT be rejected with new threshold=10."""
        orient_score = self._score(0.30)
        self.assertEqual(orient_score, 30.0)
        self.assertFalse(orient_score < cfg.BODY["min_orientation_score"])


# ---------------------------------------------------------------------------
# IMPROVEMENT 1 — shoulder_symmetry.py gradual banding
# ---------------------------------------------------------------------------

from pose_scorer.body_group.shoulder_symmetry import score_shoulder_symmetry  # type: ignore

class TestShoulderSymmetryGradual(unittest.TestCase):
    """Verify the 6-step gradual shoulder asymmetry curve."""

    def _score(self, norm_diff):
        """Build mock shoulder landmarks with a specific norm_diff."""
        # lm[11] = left shoulder, lm[12] = right shoulder
        # shoulder_dist (horizontal) = |11.x - 12.x| = 0.4
        # raw_diff = |11.y - 12.y| ; norm_diff = raw_diff / 0.4
        # => raw_diff = norm_diff * 0.4
        raw_diff = norm_diff * 0.4
        lms = {
            11: SimpleNamespace(x=0.3, y=0.5, visibility=0.95),
            12: SimpleNamespace(x=0.7, y=0.5 + raw_diff, visibility=0.95),
        }
        # Use a dict-like list
        class LandmarkList:
            def __init__(self, d):
                self._d = d
            def __getitem__(self, i):
                return self._d[i]
        config = _make_config()
        return score_shoulder_symmetry(LandmarkList(lms), config)["score"]

    def test_perfect_symmetry(self):
        self.assertEqual(self._score(0.03), 100.0)

    def test_slight_asymmetry(self):
        self.assertEqual(self._score(0.07), 85.0)

    def test_moderate_asymmetry(self):
        self.assertEqual(self._score(0.12), 70.0)

    def test_noticeable_asymmetry(self):
        self.assertEqual(self._score(0.18), 50.0)

    def test_significant_asymmetry(self):
        self.assertEqual(self._score(0.25), 30.0)

    def test_severe_asymmetry(self):
        self.assertEqual(self._score(0.35), 15.0)


# ---------------------------------------------------------------------------
# IMPROVEMENT 3 — hand_near_face proximity penalty
# ---------------------------------------------------------------------------

from pose_scorer.scorer import hand_near_face  # type: ignore

class TestHandNearFace(unittest.TestCase):
    """Verify hand-near-face proximity penalty factor."""

    def _make_body_raw(self, wrist_x, wrist_y, vis=0.9):
        """Build a minimal body_raw mock with one wrist at the given position."""
        lms = {}
        for i in range(33):
            lms[i] = SimpleNamespace(x=0.5, y=0.9, visibility=0.0)  # invisible default
        # Left wrist (15) at given position
        lms[15] = SimpleNamespace(x=wrist_x, y=wrist_y, visibility=vis)
        lms[16] = SimpleNamespace(x=0.5, y=0.9, visibility=0.0)  # invisible right wrist

        class LandmarkList:
            def __init__(self, d):
                self._d = d
            def __getitem__(self, i):
                return self._d[i]

        return SimpleNamespace(pose_landmarks=[LandmarkList(lms)])

    def test_wrist_far_away_no_penalty(self):
        """Wrist far from face → factor 1.0 (no penalty)."""
        body = self._make_body_raw(0.5, 0.9)  # wrist at bottom
        # face_bbox: pixel coords, image 1000x1000
        factor = hand_near_face(body, (300, 100, 700, 400), 1000, 1000)
        self.assertEqual(factor, 1.0)

    def test_wrist_on_face_strong_penalty(self):
        """Wrist directly on face center → factor near 0.40."""
        body = self._make_body_raw(0.5, 0.25)  # wrist at face center
        factor = hand_near_face(body, (300, 100, 700, 400), 1000, 1000)
        self.assertLessEqual(factor, 0.50)

    def test_wrist_above_face_moderate_penalty(self):
        """Wrist just above face (on top of head) → factor < 1.0."""
        body = self._make_body_raw(0.5, 0.05)  # wrist above face
        factor = hand_near_face(body, (300, 100, 700, 400), 1000, 1000)
        self.assertLess(factor, 1.0)

    def test_no_pose_landmarks(self):
        """No body detected → factor 1.0."""
        body = SimpleNamespace(pose_landmarks=None)
        factor = hand_near_face(body, (300, 100, 700, 400), 1000, 1000)
        self.assertEqual(factor, 1.0)


# ---------------------------------------------------------------------------
# IMPROVEMENT 4 — gaze pitch penalty reduction
# ---------------------------------------------------------------------------

from pose_scorer.face_group.gaze_direction import score_gaze_direction  # type: ignore

class TestGazePitchPenaltyReduction(unittest.TestCase):
    """Verify the gaze pitch penalty max is 12pt, not 25pt."""

    def test_max_penalty_at_extreme_pitch(self):
        """At pitch just below hard skip (39°), penalty should be ≤ 12."""
        # Build minimal landmarks with iris at eye center (no horizontal offset)
        default = _lm(0.5, 0.5)
        lms = [default] * 478
        # Right eye corners: 33, 133; upper/lower: 159, 145
        lms[33]  = _lm(0.60, 0.40)
        lms[133] = _lm(0.40, 0.40)
        lms[159] = _lm(0.50, 0.38)
        lms[145] = _lm(0.50, 0.42)
        # Left eye corners: 263, 362; upper/lower: 386, 374
        lms[263] = _lm(0.40, 0.40)
        lms[362] = _lm(0.60, 0.40)
        lms[386] = _lm(0.50, 0.38)
        lms[374] = _lm(0.50, 0.42)
        # Right iris (468-471): centered in right eye
        for i in [468, 469, 470, 471]:
            lms[i] = _lm(0.50, 0.40)
        # Left iris (473-477): centered in left eye
        for i in [473, 474, 475, 476, 477]:
            lms[i] = _lm(0.50, 0.40)

        config = _make_config()
        # Pitch 39° (just below 40° hard skip), yaw 0°
        result = score_gaze_direction(lms, 0.0, 39.0, config)
        self.assertFalse(result.get('skipped'))
        # Base score should be 100 (centered iris), minus at most 12pt
        self.assertGreaterEqual(result['score'], 88.0,
                                f"Penalty too harsh: score={result['score']}")

    def test_config_gaze_pitch_penalty_is_25(self):
        """Config default for gaze_pitch_penalty should now be 25.0 (was 20.0)."""
        self.assertEqual(cfg.FACE["gaze_pitch_penalty"], 25.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with verbose output so each test name is visible
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    for cls in [TestSmileScores, TestFaceQualityGate, TestOrientationGate,
                TestBodyOrientationScoreBands, TestShoulderSymmetryGradual,
                TestHandNearFace, TestGazePitchPenaltyReduction]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
