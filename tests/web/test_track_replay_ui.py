"""Static contract checks for the track replay and recording controls."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TrackReplayUiContractTests(unittest.TestCase):
    def test_replay_ui_exposes_required_modes_speeds_filters_and_tuning(self):
        html = (ROOT / "static" / "tracks.html").read_text(encoding="utf-8")
        script = (ROOT / "static" / "tracks.js").read_text(encoding="utf-8")

        for speed in ("0.1", "0.25", "0.5", "0.75", "1", "1.5", "2"):
            self.assertIn(f'value="{speed}"', html)
        for control_id in (
            "recordingSelect", "trackSelect", "pauseButton", "mode2d", "mode3d",
            "confirmHits", "gateDistance", "maxMisses", "deleteAfter",
            "processNoise", "confidenceDecay", "reprocessButton",
            "deleteRecordingButton",
        ):
            self.assertIn(f'id="{control_id}"', html)
        self.assertIn("const clamp =", script)
        self.assertIn("function validPoint", script)
        self.assertIn("function recordingImageSize", script)
        self.assertIn("parameters.image_width", script)
        self.assertIn("parameters.image_height", script)
        self.assertNotIn("const imageW=640,imageH=480", script)
        self.assertIn("const replayFps =", script)
        self.assertIn("playbackStartIndex", script)

    def test_main_ui_exposes_record_toggle_and_replay_link(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

        self.assertIn('id="recordToggleButton"', html)
        self.assertIn("toggleRecording()", html)
        self.assertNotIn('id="startRecordingButton"', html)
        self.assertNotIn('id="stopRecordingButton"', html)
        self.assertIn('href="/tracks"', html)
        self.assertIn("/api/track-recordings/${action}", html)
        self.assertIn("/static/vendor/three.module.js", html)
        self.assertTrue((ROOT / "static" / "vendor" / "three.module.js").is_file())


if __name__ == "__main__":
    unittest.main()
