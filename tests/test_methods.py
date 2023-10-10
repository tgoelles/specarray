from __future__ import annotations

import pytest
from pytest_check import check
from specarray import SpecArray
from xarray import DataArray


def test_repr(testdata_specim: SpecArray):
    repr_res = testdata_specim.__repr__()
    check.is_true(repr_res.startswith("SpecArray"))
    check.is_true("wavelength" in repr_res)
    check.is_true("capture" in repr_res)


def test_len(testdata_specim: SpecArray):
    check.equal(len(testdata_specim), 2)


def test_getitem(testdata_specim: SpecArray):
    check.equal(testdata_specim[0].shape, (448, 1024))
    check.equal(testdata_specim[1].shape, (448, 1024))
    check.is_instance(testdata_specim[0], DataArray)


def test_shape(testdata_specim: SpecArray):
    check.equal(testdata_specim.shape, (2, 448, 1024))


def test_dims(testdata_specim: SpecArray):
    check.equal(testdata_specim.capture.dims, ("sample", "wavelength", "point"))


def test_spectral_albedo(testdata_specim: SpecArray):
    spectral_albedo = testdata_specim.spectral_albedo
    check.is_instance(spectral_albedo, DataArray)
    check.equal(spectral_albedo.shape, testdata_specim.capture.shape)
    check.equal(spectral_albedo.name, "spectral albedo")
    check.is_true(spectral_albedo.coords["wavelength"].equals(testdata_specim.capture.coords["wavelength"]))
    check.greater_equal(spectral_albedo.min(), 0.0)
    check.less_equal(spectral_albedo.min(), 1.0)
    check.equal(spectral_albedo.dims, ("sample", "wavelength", "point"))
    check.equal(spectral_albedo.dims, testdata_specim.capture.dims)
    check.almost_equal(float(spectral_albedo.sel(sample=0).max().as_numpy().values), 0.7786328655500226)
    check.almost_equal(float(spectral_albedo.sel(sample=0).min().as_numpy().values), 0.125)
    check.almost_equal(
        float(spectral_albedo.sel(sample=0, point=0).sel(wavelength=1000, method="nearest").as_numpy().values),
        0.28365385,
    )


def test_broadband_albedo(testdata_specim: SpecArray):
    broadband_albedo = testdata_specim.broadband_albedo
    check.is_instance(broadband_albedo, DataArray)
    check.equal(broadband_albedo.shape, (2, 1024))
    check.equal(broadband_albedo.dims, ("sample", "point"))
    check.equal(broadband_albedo.name, "broadband albedo")
    check.almost_equal(float(broadband_albedo.sel(sample=0, point=0).as_numpy().values), 0.61686938)


@pytest.mark.parametrize("test_sets", ["testdata_no_black", "testdata_no_white"], indirect=True)
def test_spectal_albedo_no_black_or_white(test_sets: SpecArray):
    with pytest.raises(ValueError):
        test_sets.spectral_albedo


@pytest.mark.parametrize("test_sets", ["testdata_no_black", "testdata_no_white"], indirect=True)
def test_broadband_albedo_no_black_or_white(test_sets: SpecArray):
    with pytest.raises(ValueError):
        test_sets.broadband_albedo
