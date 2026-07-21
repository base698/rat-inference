"""Tests for shared YOLO inference helpers."""

from __future__ import annotations

import unittest

from ratbot.vision.yolo_inference import target_class_matches


class YoloInferenceHelperTests(unittest.TestCase):
    def test_class_zero_fallback_does_not_match_coco_person(self):
        self.assertFalse(target_class_matches("person", 0, "class0"))
        self.assertFalse(target_class_matches("person", 0, "class_0"))

    def test_class_zero_fallback_matches_generated_model_label(self):
        self.assertTrue(target_class_matches("class0", 0, "class0"))
        self.assertTrue(target_class_matches("class_0", 0, "class0"))

    def test_numeric_target_still_matches_numeric_class_id(self):
        self.assertTrue(target_class_matches("person", 0, "0"))

    def test_named_targets_match_named_model_classes(self):
        self.assertTrue(target_class_matches("bottle", 39, "bottle"))
        self.assertTrue(target_class_matches("coffee cup", 41, "cup"))


if __name__ == "__main__":
    unittest.main()
