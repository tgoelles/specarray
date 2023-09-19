# from the spectral pyhton package

from pathlib import Path
import shutil
import numpy as np
import spectral.io.envi as envi
import spectral as spy
from numpy.testing import assert_almost_equal


testdatadir = Path(__file__).parent.absolute() / "specim_data" / "capture"
specim_ex_name = "fake"


def test_save_image_ndarray(prefix=""):
    """Test saving an ENVI formated image from a numpy.ndarray."""
    (R, B, C) = (10, 20, 30)
    (r, b, c) = (3, 8, 23)
    datum = 33
    data = np.zeros((R, B, C), dtype=np.uint16)
    data[r, b, c] = datum
    fname = testdatadir / f"{prefix}{specim_ex_name}.hdr"
    envi.save_image(fname, data, interleave="bil")
    img = spy.open_image(fname)
    assert_almost_equal(img[r, b, c], datum)


def main():
    shutil.rmtree(testdatadir, ignore_errors=True)
    testdatadir.mkdir(parents=True, exist_ok=True)
    test_save_image_ndarray()
    test_save_image_ndarray("DARKREF_")
    test_save_image_ndarray("WHITEREF_")


if __name__ == "__main__":
    main()
