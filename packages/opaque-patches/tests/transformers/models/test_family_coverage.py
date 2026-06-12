# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Guard: every registered model family has a dedicated ``test_<family>.py``.

The repo convention is one test file per family. This fails if a family is
registered without a matching test file, so a new family can't be added without
test coverage.
"""

import glob
import os

import pytest

pytest.importorskip("transformers")

from opaque.api.patches.transformers import supported_families


def test_every_registered_family_has_a_test_file():
    here = os.path.dirname(__file__)
    test_files = {
        os.path.basename(p)[len("test_"):-len(".py")]
        for p in glob.glob(os.path.join(here, "test_*.py"))
    }
    missing = sorted(set(supported_families()) - test_files)
    assert not missing, (
        f"registered families without a test_<family>.py: {missing}"
    )
