import shutil
from os import remove
from pathlib import Path

import pandas as pd
import pytest
import spectral.io.envi as envi
import xarray as xr
from specarray import SpecArray

thisdir = Path(__file__).parent.absolute()
testdatadir = thisdir / "testdata"
specim_data_dir = thisdir / "testdata" / "specim_data"
no_black_dir = testdatadir / "no_black"
no_white_dir = testdatadir / "no_white"
testdatadir1 = specim_data_dir / "capture"
specim_ex_name = "fake"


def _save_image_ndarray(prefix=""):
    """Test saving an ENVI formated image from a numpy.ndarray."""
    wavelengths = pd.read_csv(testdatadir / "wavelengths.csv")
    wavelengths = list(wavelengths["wavelengths (nm)"].values)
    if prefix == "":
        data = xr.open_dataset(testdatadir / "capture.nc")["capture"]
    elif prefix == "DARKREF_":
        data = xr.open_dataset(testdatadir / "black.nc")["black"]
    elif prefix == "WHITEREF_":
        data = xr.open_dataset(testdatadir / "white.nc")["white"]
    fname = testdatadir1 / f"{prefix}{specim_ex_name}.hdr"
    envi.save_image(fname, data.as_numpy().values, interleave="bil", metadata=dict(wavelength=wavelengths))


def gen_data():
    shutil.rmtree(specim_data_dir, ignore_errors=True)
    shutil.rmtree(no_black_dir, ignore_errors=True)
    shutil.rmtree(no_white_dir, ignore_errors=True)
    testdatadir1.mkdir(parents=True, exist_ok=True)
    _save_image_ndarray()
    _save_image_ndarray("DARKREF_")
    _save_image_ndarray("WHITEREF_")
    # rename all files from .img to .raw
    for f in testdatadir1.glob("*.img"):
        f.rename(f.with_suffix(".raw"))
    # copy folder
    shutil.copytree(specim_data_dir, no_black_dir)
    shutil.copytree(specim_data_dir, no_white_dir)
    remove(no_black_dir / "capture" / "DARKREF_fake.hdr")
    remove(no_black_dir / "capture" / "DARKREF_fake.raw")
    remove(no_white_dir / "capture" / "WHITEREF_fake.raw")
    remove(no_white_dir / "capture" / "WHITEREF_fake.hdr")


gen_data()


@pytest.fixture()
def testdata_specim_folder() -> Path:
    return specim_data_dir


@pytest.fixture()
def testdata_no_black_folder() -> Path:
    return no_black_dir


@pytest.fixture()
def testdata_no_white_folder() -> Path:
    return no_white_dir


@pytest.fixture()
def testdata_specim(testdata_specim_folder):
    return SpecArray.from_folder(testdata_specim_folder)


@pytest.fixture()
def testdata_no_black(testdata_no_black_folder):
    return SpecArray.from_folder(testdata_no_black_folder)


@pytest.fixture()
def testdata_no_white(testdata_no_white_folder):
    return SpecArray.from_folder(testdata_no_white_folder)


@pytest.fixture
def test_sets(request):
    """for testing with different datasets"""
    return request.getfixturevalue(request.param)
