from specarray import SpecArray

from pytest_check import check


def test_from_dir(testdata_specim_folder):
    spec = SpecArray.from_folder(testdata_specim_folder)
    check.equal(spec.shape, (2, 1024, 448))
