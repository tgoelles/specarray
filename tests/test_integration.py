from specarray import SpecArray

from pytest_check import check


def test_from_dir(testdata_specim: SpecArray):
    check.equal(testdata_specim.shape, (2, 1024, 448))
