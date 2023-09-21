import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from pytest_check import check

from specarray.specarray import SpecArray


def test_specarray():
    # Create a SpecArray object from a folder
    folder = Path("/path/to/folder")
    specarray = SpecArray.from_folder(folder)

    # Test the __len__ method
    check(len(specarray) == specarray.capture.shape[0])

    # Test the __getitem__ method
    check(np.array_equal(specarray[0], specarray.capture[0]))

    # Test the shape property
    check(specarray.shape == specarray.capture.shape)

    # Test the broadband_albedo property
    broadband_albedo = specarray.broadband_albedo
    check(isinstance(broadband_albedo, xr.DataArray))
    check(broadband_albedo.shape == (specarray.capture.shape[0], specarray.capture.shape[2]))
    check(broadband_albedo.name == "broadband_albedo")

    # Test the spectral_albedo property
    spectral_albedo = specarray.spectral_albedo
    check(isinstance(spectral_albedo, xr.DataArray))
    check(spectral_albedo.shape == specarray.capture.shape)
    check(spectral_albedo.name == "capture")
    check(spectral_albedo.coords["wavelength"].equals(specarray.capture.coords["wavelength"]))
    check(spectral_albedo.min() >= 0.0)
    check(spectral_albedo.max() <= 1.0)

    # Test the _gen_wavelength_point_df method
    raw_array = np.random.rand(specarray.capture.shape[0], specarray.capture.shape[2])
    df = specarray._gen_wavelength_point_df(raw_array)
    check(isinstance(df, pd.DataFrame))
    check(df.shape == (len(specarray.wavelengths), specarray.capture.shape[2]))
    check(df.index.equals(specarray.wavelengths))
