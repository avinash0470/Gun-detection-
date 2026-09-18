import os
import json
import shutil
import numpy as np
import unittest

from pipeline.config import SystemConfig, TemporalConfig, QualityGateConfig
from pipeline.core import GunDetectionPipeline
from pipeline.temporal import TemporalValidator
from pipeline.quality_gate import FrameQualityGate
from pipeline.alert_system import AlertSystem


class TestGunDetectionPipeline(unittest.TestCase):
    def setUp(self):
        self.config = SystemConfig()
        self.pipeline = GunDetectionPipeline(self.config)

    def test_sticky_suspect_concealment(self):
        """Issue 1: Test that weapon concealment keeps sticky suspect status in frame 2."""
        frame_1 = {
            "brightness": 0.7, "blur": 120.0, "occlusion": 0.0,
            "people": [{"bbox": (100, 100, 200, 400)}],
            "detections_primary": [{"bbox": (110, 180, 140, 210), "confidence": 0.92}],
            "detections_secondary": [{"bbox": (108, 178, 142, 212), "confidence": 0.88}],
            "crop_sim_info": {"sim_firearm": 0.95, "sim_negative": 0.05, "matched_category": "handgun"},
            "pose_context": "aimed",
            "sim_duration_sec": 4.2
        }
        out_1 = self.pipeline.process_frame(frame_1, location_risk=0.8)
        p1 = out_1["persons"][0]
        self.assertTrue(p1["is_suspect"])
        self.assertTrue(p1["previously_armed"])
        self.assertFalse(p1["is_concealed"])

        frame_2 = {
            "brightness": 0.7, "blur": 120.0, "occlusion": 0.0,
            "people": [{"bbox": (100, 100, 200, 400)}],
            "detections_primary": [],
            "detections_secondary": [],
            "pose_context": "casual"
        }
        out_2 = self.pipeline.process_frame(frame_2, location_risk=0.8)
        p2 = out_2["persons"][0]
        self.assertTrue(p2["is_suspect"])
        self.assertTrue(p2["previously_armed"])
        self.assertTrue(p2["is_concealed"])

    def test_alert_system_evidence_and_logging(self):
        """Issue 2: Test evidence snapshot saving and jsonl event logging."""
        test_evidence = "test_evidence_tmp"
        test_log_dir = "test_logs_tmp"
        if os.path.exists(test_evidence):
            shutil.rmtree(test_evidence)
        if os.path.exists(test_log_dir):
            shutil.rmtree(test_log_dir)

        try:
            alerts = AlertSystem(evidence_dir=test_evidence, log_dir=test_log_dir)
            frame_arr = np.ones((200, 200, 3), dtype=np.uint8) * 128
            frame_other = np.zeros((200, 200, 3), dtype=np.uint8)

            hash1 = alerts._generate_evidence_hash("t1", 0.9, 1000.0, frame_arr)
            hash2 = alerts._generate_evidence_hash("t1", 0.9, 1000.0, frame_other)
            self.assertNotEqual(hash1, hash2)

            event_danger = alerts.dispatch("threat_1", {"level": "DANGER", "score": 0.95}, frame_arr)
            self.assertIsNotNone(event_danger["snapshot_path"])
            self.assertTrue(os.path.exists(event_danger["snapshot_path"]))

            event_low = alerts.dispatch("threat_2", {"level": "LOW", "score": 0.20}, frame_arr)
            self.assertEqual(event_low["action_taken"], "LOGGED_ONLY")

            log_file = os.path.join(test_log_dir, "detection_events.jsonl")
            self.assertTrue(os.path.exists(log_file))
            with open(log_file, "r") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
        finally:
            if os.path.exists(test_evidence):
                shutil.rmtree(test_evidence)
            if os.path.exists(test_log_dir):
                shutil.rmtree(test_log_dir)

    def test_temporal_validator_absent_purge(self):
        """Issue 4: Test that tracks dropped from active_track_ids are purged after absent timeout."""
        tv_config = TemporalConfig(history_frames=4, min_detections_in_window=1, occlusion_recovery_frames=5)
        tv = TemporalValidator(tv_config)

        # Track person_a with active detections
        for _ in range(3):
            tv.validate(["person_a"], ["person_a"])

        self.assertIn("person_a", tv.track_history)
        self.assertEqual(tv.occlusion_counters["person_a"], 0)

        # Track vanishes from active_track_ids for 6 frames (> 5)
        for _ in range(6):
            tv.validate([], [])

        self.assertNotIn("person_a", tv.track_history)
        self.assertNotIn("person_a", tv.occlusion_counters)
        self.assertNotIn("person_a", tv.gun_first_seen_timestamp)
        self.assertNotIn("person_a", tv.absent_counters)

    def test_quality_gate_occlusion(self):
        """Issue 5: Test real numpy frame occlusion metric calculation."""
        qg = FrameQualityGate(QualityGateConfig(occlusion_threshold=0.5))
        clear_frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        res_clear = qg.process(clear_frame)
        self.assertEqual(res_clear["occlusion_score"], 0.0)
        self.assertFalse(res_clear["is_occluded"])

        covered_frame = clear_frame.copy()
        covered_frame[:, :400] = 0  # ~62% covered
        res_covered = qg.process(covered_frame)
        self.assertGreaterEqual(res_covered["occlusion_score"], 0.5)
        self.assertTrue(res_covered["is_occluded"])

    def test_location_risk_weight(self):
        """Issue 7: Test that location_risk proportionally influences the final risk score."""
        frame = {
            "brightness": 0.7, "blur": 120.0, "occlusion": 0.0,
            "people": [{"bbox": (100, 100, 200, 400)}],
            "detections_primary": [{"bbox": (110, 180, 140, 210), "confidence": 0.90}],
            "detections_secondary": [{"bbox": (108, 178, 142, 212), "confidence": 0.88}],
            "crop_sim_info": {"sim_firearm": 0.95, "sim_negative": 0.05, "matched_category": "handgun"},
            "pose_context": "aimed",
            "sim_duration_sec": 4.2
        }
        out_low = self.pipeline.process_frame(frame, location_risk=0.0)
        score_low = out_low["detections"][0]["risk_score"]

        out_high = self.pipeline.process_frame(frame, location_risk=0.8)
        score_high = out_high["detections"][0]["risk_score"]

        self.assertGreater(score_high, score_low)
        self.assertAlmostEqual(score_high - score_low, 0.8 * 0.10, places=4)

    def test_detect_gun_without_person(self):
        """Test require_person toggle: detecting guns when no person is present vs requiring person."""
        # Frame with gun on table / wall, NO people
        frame_no_person = {
            "brightness": 0.7, "blur": 120.0, "occlusion": 0.0,
            "people": [],
            "detections_primary": [{"bbox": (300, 300, 350, 350), "confidence": 0.88}],
            "crop_sim_info": {"sim_firearm": 0.90, "sim_negative": 0.05, "matched_category": "handgun"},
            "pose_context": "unattended"
        }

        # Case 1: require_person = True -> Should discard gun detection
        self.pipeline.config.detector.require_person = True
        out_req = self.pipeline.process_frame(frame_no_person)
        self.assertEqual(len(out_req["detections"]), 0)

        # Case 2: require_person = False -> Should detect unattended gun
        self.pipeline.config.detector.require_person = False
        out_unattended = self.pipeline.process_frame(frame_no_person)
        self.assertEqual(len(out_unattended["detections"]), 1)
        self.assertTrue(out_unattended["detections"][0]["track_id"].startswith("gun_unassociated"))


if __name__ == "__main__":
    unittest.main()
