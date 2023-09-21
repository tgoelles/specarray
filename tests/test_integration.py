from specarray import SpecArray
from pathlib import Path

from pytest_check import check
import pytest


def test_from_dir(testdata_specim: SpecArray):
    check.equal(testdata_specim.shape, (2, 1024, 448))


def test_no_black(testdata_no_black: Path):
    with pytest.warns(UserWarning):
        no_black = SpecArray.from_folder(testdata_no_black)
    check.is_instance(no_black, SpecArray)
    check.equal(no_black.shape, (2, 1024, 448))
    check.equal(no_black.black.shape, ())


def test_no_white(testdata_no_white: Path):
    with pytest.warns(UserWarning):
        no_white = SpecArray.from_folder(testdata_no_white)
    check.is_instance(no_white, SpecArray)
    check.equal(no_white.shape, (2, 1024, 448))
    check.equal(no_white.white.shape, ())
