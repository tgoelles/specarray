from __future__ import annotations

import pytest
from pytest_check import check
from specarray import SpecArray
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


def test_spectral_albedo(testdata_specim: SpecArray):
    spectral_albedo = testdata_specim.spectral_albedo

    check.is_instance(spectral_albedo, DataArray)
    check.equal(spectral_albedo.shape, testdata_specim.capture.shape)
    check.equal(spectral_albedo.name, "spectral albedo")
    check.is_true(spectral_albedo.coords["wavelength"].equals(testdata_specim.capture.coords["wavelength"]))
    check.greater_equal(spectral_albedo.min(), 0.0)
    check.less_equal(spectral_albedo.min(), 1.0)
    # single_value_min = float(spectral_albedo[0][0].min().compute())
    # single_value_max = float(spectral_albedo[0][0].max().compute())
    # check.almost_equal(single_value_min, 0.2482758620689655)
    # check.almost_equal(single_value_max, 0.7558048525958779)


def test_broadband_albedo(testdata_specim: SpecArray):
    broadband_albedo = testdata_specim.broadband_albedo
    check.is_instance(broadband_albedo, DataArray)
    check.equal(broadband_albedo.shape, (2, 1024))
    check.equal(broadband_albedo.name, "broadband albedo")


@pytest.mark.parametrize("test_sets", ["testdata_no_black", "testdata_no_white"], indirect=True)
def test_spectal_albedo_no_black_or_white(test_sets: SpecArray):
    with pytest.raises(ValueError):
        test_sets.spectral_albedo


@pytest.mark.parametrize("test_sets", ["testdata_no_black", "testdata_no_white"], indirect=True)
def test_spectal_albedo_no_black_or_white(test_sets: SpecArray):
    with pytest.raises(ValueError):
        test_sets.broadband_albedo
