from pathlib import Path

import pytest
from specarray import SpecArray


@pytest.fixture()
def testdata_specim_folder() -> Path:
    return Path(__file__).parent.absolute() / "testdata/specim_data"


@pytest.fixture()
def testdata_specim(testdata_specim_folder):
    return SpecArray.from_folder(testdata_specim_folder)
