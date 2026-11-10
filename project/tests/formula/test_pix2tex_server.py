import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.analysis.formula import formula_recognizer
from src.analysis.formula.formula_recognizer import recognize_with_isolated_pix2tex


class Pix2texServerTests(unittest.TestCase):
    def tearDown(self):
        process = formula_recognizer._pix2tex_server["process"]
        if process is not None:
            process.kill()
            formula_recognizer._pix2tex_server["process"] = None

    def test_one_worker_process_serves_every_formula(self):
        # The main venv has no pix2tex, so the worker answers with an error
        # payload: recognition falls back (None) but the process is reused.
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"PIX2TEX_PYTHON": sys.executable}):
            images = [Path(tmp) / f"f{i}.png" for i in range(2)]
            for image in images:
                image.write_bytes(b"not really a png")

            self.assertIsNone(recognize_with_isolated_pix2tex(images[0]))
            first = formula_recognizer._pix2tex_server["process"]
            self.assertIsNone(recognize_with_isolated_pix2tex(images[1]))

            self.assertIsNotNone(first)
            self.assertIs(formula_recognizer._pix2tex_server["process"], first)
            self.assertIsNone(first.poll())


if __name__ == "__main__":
    unittest.main()
