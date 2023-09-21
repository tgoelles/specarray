from pathlib import Path

import pytest
from specarray import SpecArray

from pathlib import Path
import shutil

import pandas as pd
import spectral.io.envi as envi
import xarray as xr


thisdir = Path(__file__).parent.absolute()
testdatadir = thisdir / "testdata" / "specim_data" / "capture"
specim_ex_name = "fake"


def _save_image_ndarray(prefix=""):
    """Test saving an ENVI formated image from a numpy.ndarray."""
    wavelengths = pd.read_csv(thisdir / "wavelengths.csv")
    wavelengths = list(wavelengths["wavelengths (nm)"].values)
    if prefix == "":
        data = xr.open_dataset("/workspaces/specarray/tests/testdata/capture.nc")["capture"]
    elif prefix == "DARKREF_":
        data = xr.open_dataset("/workspaces/specarray/tests/testdata/black.nc")["black"]
    elif prefix == "WHITEREF_":
        data = xr.open_dataset("/workspaces/specarray/tests/testdata/white.nc")["white"]
    fname = testdatadir / f"{prefix}{specim_ex_name}.hdr"
    envi.save_image(fname, data.as_numpy().values, interleave="bil", metadata=dict(wavelength=wavelengths))


def gen_data():
    shutil.rmtree(testdatadir, ignore_errors=True)
    testdatadir.mkdir(parents=True, exist_ok=True)
    _save_image_ndarray()
    _save_image_ndarray("DARKREF_")
    _save_image_ndarray("WHITEREF_")
    # rename all files from .img to .raw
    for f in testdatadir.glob("*.img"):
        f.rename(f.with_suffix(".raw"))
    shutil.copy(testdatadir, thisdir / "specim_data_no_darkref")


gen_data


@pytest.fixture()
def testdata_specim_folder() -> Path:
    return Path(__file__).parent.absolute() / "testdata/specim_data"


@pytest.fixture()
def testdata_specim(testdata_specim_folder):
    return SpecArray.from_folder(testdata_specim_folder)
