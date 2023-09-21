from specarray import SpecArray
from pathlib import Path

from pytest_check import check
import pytest


def test_from_dir(testdata_specim: SpecArray):
    check.equal(testdata_specim.shape, (2, 1024, 448))


@pytest.mark.parametrize("test_sets", ["testdata_no_black", "testdata_no_white"], indirect=True)
def test_no_white_or_black(test_sets: SpecArray):
    check.is_instance(test_sets, SpecArray)
    check.equal(test_sets.shape, (2, 1024, 448))
