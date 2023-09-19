# from the spectral pyhton package

from pathlib import Path
import shutil

import pandas as pd
import spectral.io.envi as envi


thisdir = Path(__file__).parent.absolute()
testdatadir = thisdir / "specim_data" / "capture"
specim_ex_name = "fake"


def test_save_image_ndarray(prefix=""):
    """Test saving an ENVI formated image from a numpy.ndarray."""
    wavelengths = pd.read_csv(thisdir / "wavelengths.csv")
    wavelengths = list(wavelengths["wavelengths (nm)"].values)
    if prefix == "":
        data = pd.read_csv(thisdir / "capture.csv").values
    elif prefix == "DARKREF_":
        data = pd.read_csv(thisdir / "black.csv").values
    elif prefix == "WHITEREF_":
        data = pd.read_csv(thisdir / "white.csv").values
    fname = testdatadir / f"{prefix}{specim_ex_name}.hdr"
    envi.save_image(fname, data, interleave="bil", metadata=dict(wavelength=wavelengths))


def main():
    shutil.rmtree(testdatadir, ignore_errors=True)
    testdatadir.mkdir(parents=True, exist_ok=True)
    test_save_image_ndarray()
    test_save_image_ndarray("DARKREF_")
    test_save_image_ndarray("WHITEREF_")
    # rename all files from .img to .raw
    for f in testdatadir.glob("*.img"):
        f.rename(f.with_suffix(".raw"))


if __name__ == "__main__":
    main()
