"""Offline tests using invented image content only."""

import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import cv2  # Keep the real module loaded across the mocked detector test.
from PIL import Image, ImageDraw, PngImagePlugin

from research.image_redaction import local_image_redactor as tool


def fake_ocr(image):
    values = [
        ("Patient", (10, 10, 60, 24), (1, 1, 1)),
        ("Ada", (65, 10, 90, 24), (1, 1, 1)),
        ("Inventova", (95, 10, 165, 24), (1, 1, 1)),
        ("Order", (10, 35, 50, 49), (1, 1, 2)),
        ("AB1234", (55, 35, 110, 49), (1, 1, 2)),
        ("Ada", (10, 60, 35, 74), (1, 1, 3)),
        ("Inventova", (40, 60, 110, 74), (1, 1, 3)),
        ("Glucose", (10, 95, 65, 109), (1, 1, 4)),
        ("5.2", (80, 95, 105, 109), (1, 1, 4)),
        ("mmol/L", (110, 95, 165, 109), (1, 1, 4)),
        ("Ada", (10, 140, 35, 154), (1, 1, 5)),
        ("Inventova", (40, 140, 110, 154), (1, 1, 5)),
    ]
    parts = []
    words = []
    offset = 0
    previous = None
    for value, box, line in values:
        if parts:
            separator = "\n" if previous != line else " "
            parts.append(separator)
            offset += len(separator)
        words.append(tool.Word(value, offset, offset + len(value), box, line))
        parts.append(value)
        offset += len(value)
        previous = line
    return "".join(parts), words


class LocalImageRedactorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.originals = self.root / "originals"
        self.originals.mkdir()
        self.source = self.originals / "invented.png"
        image = Image.new("RGB", (240, 180), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 95, 165, 109), fill=(40, 80, 120))
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Comment", "invented metadata")
        image.save(self.source, pnginfo=metadata)
        self.work = self.root / "private"
        self.approved = self.root / "approved"

    def fake_entities(self, text, **kwargs):
        first = text.index("Ada Inventova")
        last = text.rindex("Ada Inventova")
        return [{"start": first, "end": first + len("Ada Inventova"), "kind": "person"},
                {"start": last, "end": last + len("Ada Inventova"), "kind": "person"}]

    def propose(self, **kwargs):
        with patch.object(tool, "ocr_words", side_effect=fake_ocr), \
             patch.object(tool, "detect_entities", side_effect=self.fake_entities), \
             patch.object(tool, "code_boxes", return_value=([], ["Inspect codes"])):
            return tool.propose(self.source, self.work, "D0001", **kwargs)

    def test_occurrence_and_multiline_mapping(self):
        text, words = fake_ocr(None)
        first = text.index("Ada Inventova")
        repeated = text.rindex("Ada Inventova")
        boxes = tool.spans_to_boxes(words, first, first + 13, (240, 180))
        self.assertEqual(boxes, [[62, 7, 168, 27]])
        self.assertEqual(tool.spans_to_boxes(words, repeated, repeated + 13, (240, 180)), [[7, 137, 113, 157]])
        middle = text.index("AB1234")
        multiline = tool.spans_to_boxes(words, middle, text.index("Inventova", middle) + 9, (240, 180))
        self.assertEqual(len(multiline), 2)

    def test_private_candidate_pixel_coverage_and_approval(self):
        original_hash = tool.digest(self.source)
        record = self.propose()
        candidate = self.work / "D0001" / "v001" / "candidate.png"
        self.assertFalse(self.approved.exists())
        self.assertEqual(original_hash, tool.digest(self.source))
        with Image.open(candidate) as image:
            self.assertEqual(image.getpixel((90, 17)), (0, 0, 0))
            self.assertEqual(image.getpixel((57, 42)), (0, 0, 0))
            self.assertEqual(image.getpixel((50, 145)), (0, 0, 0))
            self.assertEqual(image.getpixel((90, 100)), (40, 80, 120))
            self.assertEqual(image.info, {})
        with self.assertRaises(tool.RedactionError):
            tool.decide(self.work, self.approved, "D0001", 1, "0" * 64, "keep")
        self.assertFalse(self.approved.exists())
        decision = tool.decide(self.work, self.approved, "D0001", 1, record["candidate_sha256"], "keep",
                               codes={"organization_code": "LAB_A", "document_status": "completed"})
        exported = self.approved / decision["export_name"]
        self.assertEqual(tool.digest(exported), record["candidate_sha256"])
        manifest = json.loads(exported.with_suffix(".json").read_text())
        self.assertEqual(manifest["export_sha256"], record["candidate_sha256"])
        self.assertNotIn("source_path", manifest)
        self.assertNotIn("invented metadata", json.dumps(manifest))
        self.assertTrue(tool.approval_valid(self.approved, decision))
        with self.assertRaises(tool.RedactionError):
            tool.decide(self.work, self.approved, "D0001", 1, record["candidate_sha256"], "keep")
        exported.write_bytes(b"changed")
        self.assertFalse(tool.approval_valid(self.approved, decision))

    def test_revision_preserves_prior_candidate_and_requires_new_approval(self):
        first = self.propose()
        original_candidate = self.work / "D0001" / "v001" / "candidate.png"
        first_hash = tool.digest(original_candidate)
        second = self.propose(manual=[[170, 140, 200, 170]])
        self.assertEqual(tool.digest(original_candidate), first_hash)
        self.assertNotEqual(first["candidate_sha256"], second["candidate_sha256"])
        self.assertEqual(second["version"], 2)
        self.assertEqual(second["masks"][-1]["origin"], "manual")
        self.assertFalse(self.approved.exists())

    def test_revision_can_reuse_private_source_mapping(self):
        self.propose()
        with patch.object(tool, "ocr_words", side_effect=fake_ocr), \
             patch.object(tool, "detect_entities", side_effect=self.fake_entities), \
             patch.object(tool, "code_boxes", return_value=([], ["Inspect codes"])), \
             redirect_stdout(io.StringIO()):
            status = tool.main(["revise", "--work", str(self.work), "--id", "D0001", "--mask",
                                "170", "140", "200", "170"])
        self.assertEqual(status, 0)
        _, record = tool.private_record(self.work, "D0001", 2)
        self.assertEqual(record["masks"][-1]["box"], [170, 140, 200, 170])

    def test_manual_revision_reuses_detection_without_ocr_or_model(self):
        first = self.propose()
        with patch.object(tool, "ocr_words", side_effect=AssertionError("OCR should not run")), \
             patch.object(tool, "detect_entities", side_effect=AssertionError("model should not run")), \
             patch.object(tool, "code_boxes", side_effect=AssertionError("code detection should not run")):
            second = tool.propose(self.source, self.work, "D0001", manual=[[170, 140, 200, 170]])
        original_automatic = [mask for mask in first["masks"] if mask["origin"] == "automatic"]
        revised_automatic = [mask for mask in second["masks"] if mask["origin"] == "automatic"]
        self.assertEqual(original_automatic, revised_automatic)
        self.assertEqual(second["masks"][-1]["box"], [170, 140, 200, 170])

    def test_false_mask_can_be_removed_without_rerunning_detector(self):
        first = self.propose()
        index = next(index for index, mask in enumerate(first["masks"]) if mask["box"][1] == 137)
        prior_file = self.work / "D0001" / "v001" / "candidate.png"
        prior_hash = tool.digest(prior_file)
        with patch.object(tool, "ocr_words", side_effect=AssertionError("OCR should not run")), \
             patch.object(tool, "detect_entities", side_effect=AssertionError("model should not run")):
            second = tool.propose(self.source, self.work, "D0001", drop_masks=[index])
        self.assertEqual(second["version"], 2)
        self.assertEqual(second["removed_mask_count"], 1)
        self.assertEqual(len(second["masks"]), len(first["masks"]) - 1)
        self.assertEqual(tool.digest(prior_file), prior_hash)
        self.assertFalse(self.approved.exists())
        with Image.open(self.work / "D0001" / "v002" / "candidate.png") as image:
            self.assertNotEqual(image.getpixel((50, 145)), (0, 0, 0))
            self.assertEqual(image.getpixel((57, 42)), (0, 0, 0))
            self.assertEqual(image.getpixel((90, 100)), (40, 80, 120))

    def test_invalid_drop_index_preserves_prior_candidate(self):
        self.propose()
        with patch.object(tool, "ocr_words", side_effect=AssertionError("OCR should not run")):
            with self.assertRaisesRegex(tool.RedactionError, "invalid mask index"):
                tool.propose(self.source, self.work, "D0001", drop_masks=[999])
        self.assertFalse((self.work / "D0001" / "v002").exists())

    def test_invalid_manual_rectangle_fails_before_ocr(self):
        self.propose()
        with patch.object(tool, "ocr_words", side_effect=AssertionError("OCR should not run")):
            with self.assertRaisesRegex(tool.RedactionError, "outside image"):
                tool.propose(self.source, self.work, "D0001", manual=[[170, 140, 300, 170]])
        self.assertFalse((self.work / "D0001" / "v002").exists())

    def test_skip_keeps_candidate_private(self):
        record = self.propose()
        tool.decide(self.work, self.approved, "D0001", 1, record["candidate_sha256"], "skip", "invented reason")
        self.assertFalse(self.approved.exists())

    def test_ocr_failure_and_empty_detection(self):
        with patch.object(tool, "ocr_words", side_effect=tool.RedactionError("local OCR failed")):
            with self.assertRaises(tool.RedactionError):
                tool.propose(self.source, self.work, "D0002")
        self.assertFalse((self.work / "D0002").exists())
        with patch.object(tool, "ocr_words", return_value=("", [])), \
             patch.object(tool, "detect_entities", return_value=[]), \
             patch.object(tool, "code_boxes", return_value=([], ["Inspect codes"])):
            record = tool.propose(self.source, self.work, "D0003")
        self.assertEqual(record["masks"], [])
        self.assertEqual(record["status"], "pending")

    def test_multiframe_and_unsupported(self):
        frames = [Image.new("RGB", (10, 10), "white"), Image.new("RGB", (10, 10), "black")]
        animated = self.originals / "animated.png"
        frames[0].save(animated, save_all=True, append_images=frames[1:])
        with self.assertRaisesRegex(tool.RedactionError, "multi-frame"):
            tool.load_image(animated)
        other = self.originals / "other.bmp"
        frames[0].save(other)
        with self.assertRaisesRegex(tool.RedactionError, "unsupported"):
            tool.load_image(other)
        too_wide = self.originals / "too_wide.png"
        Image.new("RGB", (tool.MAX_SIDE + 1, 1), "white").save(too_wide)
        with self.assertRaisesRegex(tool.RedactionError, "dimension limit"):
            tool.load_image(too_wide)

    def test_single_frame_webp_with_jpg_suffix_and_animated_webp(self):
        still = self.originals / "invented_mislabeled.jpg"
        Image.new("RGB", (32, 20), "white").save(still, format="WEBP")
        original_hash = tool.digest(still)
        image = tool.load_image(still)
        self.assertEqual(image.size, (32, 20))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(tool.digest(still), original_hash)
        animated = self.originals / "animated.webp"
        Image.new("RGB", (32, 20), "white").save(
            animated, format="WEBP", save_all=True,
            append_images=[Image.new("RGB", (32, 20), "black")], duration=[100, 100],
        )
        with self.assertRaisesRegex(tool.RedactionError, "multi-frame"):
            tool.load_image(animated)

    def test_exif_orientation(self):
        image = Image.new("RGB", (40, 20), "white")
        ImageDraw.Draw(image).rectangle((0, 0, 8, 8), fill=(255, 0, 0))
        exif = Image.Exif()
        exif[274] = 6
        source = self.originals / "rotated.jpg"
        image.save(source, exif=exif)
        normalized = tool.load_image(source)
        self.assertEqual(normalized.size, (20, 40))
        self.assertGreater(normalized.getpixel((16, 4))[0], 150)
        self.assertEqual(normalized.info, {})

    def test_patterns_avoid_measurement_and_collection_date(self):
        text = "Заказ № AB1234\nДата рождения: 01.02.1980\nGlucose 5.2 mmol/L\nДата взятия: 03.04.2024\nwww.lab.invalid/r/abc"
        entities = tool.pattern_entities(text)
        values = [text[entity["start"]:entity["end"]] for entity in entities]
        self.assertIn("AB1234", values)
        self.assertIn("01.02.1980", values)
        self.assertNotIn("5.2", values)
        self.assertNotIn("03.04.2024", values)
        self.assertTrue(any(value.startswith("www.") for value in values))
        self.assertTrue(tool.COLLECTION_LINE.search("Дата взятия: 03.04.2024"))
        self.assertTrue(tool.COLLECTION_LINE.search("Collection date: 03.04.2024"))

    def test_detector_is_offline_by_default_and_uses_offsets(self):
        class FakeModel:
            def to(self, device):
                self.device = device

            def predict_entities(self, text, labels, threshold):
                self.asserted = (labels, threshold)
                return [{"start": 2, "end": 5, "label": "person", "text": "Ada"}]

        model = FakeModel()
        requests = []
        module = types.SimpleNamespace(GLiNER=types.SimpleNamespace(from_pretrained=lambda identity: model))
        hub = types.SimpleNamespace(snapshot_download=lambda **kwargs: requests.append(kwargs) or "/tmp/fictional-model")
        ort = types.SimpleNamespace(disable_telemetry_events=lambda: Path(":memory:.ses").write_text("invented"))
        previous_cwd = Path.cwd()
        with patch.dict(sys.modules, {"gliner": module, "huggingface_hub": hub, "onnxruntime": ort}), \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(tool, "cached_model_dir", return_value=Path("/tmp/fictional-model")), \
             patch.object(tool, "cached_snapshot", return_value=Path("/tmp/fictional-tokenizer")), \
             patch.object(tool, "local_model_bundle", return_value=Path("/tmp/fictional-bundle")):
            result = tool.detect_entities("  Ada Inventova")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["HF_HUB_DISABLE_TELEMETRY"], "1")
        self.assertEqual(model.device, "cpu")
        self.assertEqual(Path.cwd(), previous_cwd)
        self.assertFalse((previous_cwd / ":memory:.ses").exists())
        self.assertEqual(requests, [])
        self.assertEqual(result, [{"start": 2, "end": 5, "kind": "person"}])

    def test_detector_failure_is_content_free(self):
        module = types.SimpleNamespace(GLiNER=types.SimpleNamespace(from_pretrained=lambda identity: (_ for _ in ()).throw(
            RuntimeError("secret OCR text"))))
        hub = types.SimpleNamespace(snapshot_download=lambda **kwargs: "/tmp/fictional-model")
        ort = types.SimpleNamespace(disable_telemetry_events=lambda: None)
        with patch.dict(sys.modules, {"gliner": module, "huggingface_hub": hub, "onnxruntime": ort}), \
             patch.object(tool, "cached_model_dir", return_value=Path("/tmp/fictional-model")), \
             patch.object(tool, "cached_snapshot", return_value=Path("/tmp/fictional-tokenizer")), \
             patch.object(tool, "local_model_bundle", return_value=Path("/tmp/fictional-bundle")):
            with self.assertRaisesRegex(tool.RedactionError, "local entity detector failed") as caught:
                tool.detect_entities("secret OCR text")
        self.assertNotIn("secret OCR text", str(caught.exception))

    def test_offline_model_uses_complete_cached_snapshot(self):
        cache = self.root / "hub"
        repository = cache / "models--urchade--gliner_multi-v2.1"
        snapshot = repository / "snapshots" / "fictional-revision"
        snapshot.mkdir(parents=True)
        (snapshot / "gliner_config.json").write_text("{}")
        (snapshot / "model.safetensors").write_bytes(b"fictional weights")
        (repository / "refs").mkdir()
        (repository / "refs" / "main").write_text("fictional-revision")
        with patch.dict(os.environ, {"HF_HUB_CACHE": str(cache)}):
            self.assertEqual(tool.cached_model_dir(), snapshot)

    def test_model_bundle_resolves_weights_and_tokenizer_locally(self):
        model = self.root / "model"
        tokenizer = self.root / "tokenizer"
        model.mkdir()
        tokenizer.mkdir()
        (model / "gliner_config.json").write_text('{"model_name": "microsoft/mdeberta-v3-base"}')
        (model / "model.safetensors").write_bytes(b"fictional")
        for name in tool.TOKENIZER_FILES:
            (tokenizer / name).write_bytes(b"fictional")
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            bundle = tool.local_model_bundle(model, tokenizer)
            self.assertEqual((bundle / "model.safetensors").read_bytes(), b"fictional")
            self.assertEqual((bundle / "spm.model").read_bytes(), b"fictional")
            self.assertEqual(json.loads((bundle / "gliner_config.json").read_text())["model_name"], str(bundle))
        finally:
            os.chdir(previous)

    def test_code_regions_are_masked_without_payload(self):
        image = Image.new("RGB", (100, 100), "white")
        class FakeQR:
            def detectMulti(self, pixels):
                import numpy as np
                return True, np.array([[[20, 20], [40, 20], [40, 40], [20, 40]]], dtype=float)

        class FakeCV:
            QRCodeDetector = FakeQR

        with patch.dict(sys.modules, {"cv2": FakeCV}):
            boxes, requirements = tool.code_boxes(image)
        self.assertEqual(boxes, [[15, 15, 46, 46]])
        self.assertTrue(any("Barcode detector unavailable" in item for item in requirements))
        masked = tool.render(image, [{"kind": "code", "box": boxes[0]}])
        self.assertEqual(masked.getpixel((25, 25)), (0, 0, 0))

    def test_real_code_detector_on_fictitious_qr(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures/invented_report.png"
        boxes, _ = tool.code_boxes(tool.load_image(fixture))
        self.assertTrue(any(box[0] > 750 and box[1] < 300 for box in boxes))


if __name__ == "__main__":
    unittest.main()
