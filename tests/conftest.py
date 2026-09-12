import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(autouse=True)
def _isolated_commitment_log(tmp_path, monkeypatch):
    """Every test gets its own commitment log -- never touch the real one."""
    monkeypatch.setenv("ENTROPY_COMMITMENT_LOG",
                       str(tmp_path / "commitments.jsonl"))
