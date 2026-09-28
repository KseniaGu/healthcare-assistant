"""PDF preparation checks using invented pages and mocked PII detection."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pymupdf
from PIL import Image

from research.image_redaction import local_image_redactor as tool


class PDFRedactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'originals' / 'invented.pdf'
        self.source.parent.mkdir()
        with pymupdf.open() as doc:
            for rotation in (0, 90):
                page = doc.new_page(width=144, height=216)
                page.insert_text((20, 40), 'Invented Patient')
                page.set_rotation(rotation)
            doc.set_metadata({'author': 'Fictitious private metadata'})
            doc.save(self.source)
        self.work = self.root / 'private'
        self.approved = self.root / 'approved'
        for name, result in [('ocr_words', ('', [])), ('detect_entities', []), ('code_boxes', ([], []))]:
            mock = patch.object(tool, name, return_value=result)
            mock.start()
            self.addCleanup(mock.stop)

    def test_pages_revision_and_approval(self):
        source_hash = tool.digest(self.source)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(tool.main(['propose-pdf', str(self.source), '--work', str(self.work), '--id', 'D0017']), 0)
        self.assertNotIn(str(self.source), output.getvalue())
        self.assertFalse(self.approved.exists())
        for page_id, size in [('D0017-p001', (600, 900)), ('D0017-p002', (900, 600))]:
            with Image.open(self.work / page_id / 'v001/candidate.png') as image:
                self.assertEqual(image.size, size)
                self.assertEqual(image.info, {})
        record = tool.propose(self.source, self.work, 'D0017-p002', manual=[[0, 0, 30, 30]])
        with Image.open(self.work / 'D0017-p002/v002/candidate.png') as image:
            self.assertEqual(image.getpixel((10, 10)), (0, 0, 0))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(tool.main(['revise', '--work', str(self.work), '--id', 'D0017-p002', '--drop-mask', '0']), 0)
        _, record = tool.private_record(self.work, 'D0017-p002', 3)
        with Image.open(self.work / 'D0017-p002/v003/candidate.png') as image:
            self.assertEqual(image.getpixel((10, 10)), (255, 255, 255))
        decision = tool.decide(self.work, self.approved, 'D0017-p002', 3, record['candidate_sha256'], 'keep')
        self.assertTrue(tool.approval_valid(self.approved, decision))
        manifest = json.loads((self.approved / decision['export_name']).with_suffix('.json').read_text())
        self.assertEqual(manifest['page_number'], 2)
        self.assertEqual(manifest['page_group'], 'D0017')
        self.assertEqual(manifest['source_document_id'], 'D0017')
        self.assertEqual(manifest['render_dpi'], 300)
        self.assertNotIn(str(self.source), json.dumps(manifest))
        self.assertNotIn('source_sha256', manifest)
        self.assertEqual(tool.digest(self.source), source_hash)
        (self.approved / decision['export_name']).write_bytes(b'changed')
        self.assertFalse(tool.approval_valid(self.approved, decision))
        with self.assertRaises(tool.RedactionError):
            tool.propose_pdf(self.source, self.work, 'D0017')

    def test_existing_image_export_still_works(self):
        image_path = self.source.with_suffix('.png')
        Image.new('RGB', (80, 80), 'white').save(image_path)
        record = tool.propose(image_path, self.work, 'D0020')
        decision = tool.decide(self.work, self.approved, 'D0020', 1, record['candidate_sha256'], 'keep')
        self.assertTrue(tool.approval_valid(self.approved, decision))
        manifest = json.loads((self.approved / decision['export_name']).with_suffix('.json').read_text())
        self.assertNotIn('page_number', manifest)

    def test_limits_and_changed_original(self):
        with self.assertRaises(tool.RedactionError):
            tool.load_pdf_page(self.source, 3)
        with patch.object(tool, 'MAX_PIXELS', 1), self.assertRaises(tool.RedactionError):
            tool.load_pdf_page(self.source, 1)
        tool.propose(self.source, self.work, 'D0018-p001')
        with pymupdf.open(self.source) as doc:
            doc.set_metadata({'author': 'Changed'})
            doc.saveIncr()
        with self.assertRaisesRegex(tool.RedactionError, 'different source'):
            tool.propose(self.source, self.work, 'D0018-p001', manual=[[0, 0, 10, 10]])

    def test_reject_encrypted_pdf_and_invalid_id(self):
        encrypted = self.source.with_name('encrypted.pdf')
        with pymupdf.open(self.source) as doc:
            doc.save(encrypted, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw='owner', user_pw='secret')
        with self.assertRaises(tool.RedactionError):
            tool.propose_pdf(encrypted, self.work, 'D0019')
        with self.assertRaises(tool.RedactionError):
            tool.propose_pdf(self.source, self.work, '../escape')
        self.assertFalse(self.work.exists())


if __name__ == '__main__':
    unittest.main()
