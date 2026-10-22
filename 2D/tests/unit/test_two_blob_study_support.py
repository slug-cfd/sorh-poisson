"""Checks for selecting reusable two-blob study results."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tests.two_blob_studies import common


class TwoBlobStudySupportTests(unittest.TestCase):
    def test_latest_omega_skips_newer_incomplete_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            complete = root / "omega" / "20260101-000000" / "best_omega.csv"
            partial = root / "omega" / "20260102-000000" / "best_omega.csv"
            complete.parent.mkdir(parents=True)
            partial.parent.mkdir(parents=True)
            header = "N,method,cfl,omega_selected\n"
            complete.write_text(
                header + "8,h_unsplit,0.8,0.8\n8,p_unsplit,0.8,2.0\n"
            )
            partial.write_text(header + "8,h_unsplit,0.8,0.8\n")

            required = {(8, "h_unsplit", 0.8), (8, "p_unsplit", 0.8)}
            with patch.object(common, "RESULTS_ROOT", root):
                self.assertEqual(common.latest_omega_results(required=required), complete)


if __name__ == "__main__":
    unittest.main()
