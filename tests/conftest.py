from pathlib import Path

import pytest

from .testdata.gen_data import main as gen_data_main

gen_data_main()


@pytest.fixture()
def testdata_specim_folder() -> Path:
    return Path(__file__).parent.absolute() / "testdata/specim_data"
