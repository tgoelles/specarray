from __future__ import annotations

from specarray import SpecArray
from pytest_check import check

from xarray import DataArray


def test_repr(testdata_specim: SpecArray):
    repr = testdata_specim.__repr__()
    print(repr)
    check.is_true(repr.startswith("SpecArray"))
    check.equal(len(repr), 5823)


def test_len(testdata_specim: SpecArray):
    check.equal(len(testdata_specim), 2)


def test_getitem(testdata_specim: SpecArray):
    check.equal(testdata_specim[0].shape, (1024, 448))
    check.equal(testdata_specim[1].shape, (1024, 448))
    check.is_instance(testdata_specim[0], DataArray)


def test_shape(testdata_specim: SpecArray):
    check.equal(testdata_specim.shape, (2, 1024, 448))
