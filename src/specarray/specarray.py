"""Class for Specim hyperspectral Camera data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numpy import ndarray

from .io.specim_folder import from_specim_folder


@dataclass
class SpecArray:
    """Class for Specim hyperspectral Camera data"""

    capture: xr.DataArray
    metadata: dict
    wavelengths: pd.Series

    black: xr.DataArray
    white: xr.DataArray

    @classmethod
    def from_folder(cls, folder: Path):
        """Create a SpecArray from a folder"""
        return cls(*from_specim_folder(folder))

    def __len__(self):
        """Return the number of records"""
        return len(self.capture)

    def __getitem__(self, key):
        return self.capture[key]

    @property
    def shape(self):
        """Return the shape of the data"""
        return self.capture.shape

    @property
    def spectral_albedo(self):
        """Calculate and return the spectral albedo. The values are limited to 0 and 1."""
        spectral_albedo = (self.capture - self.black.mean(dim="sample")) / (
            self.white.mean(dim="sample") - self.black.mean(dim="sample")
        )
        spectral_albedo = xr.where(spectral_albedo < 0.0, 0.0, spectral_albedo)
        spectral_albedo = xr.where(spectral_albedo > 1.0, 1.0, spectral_albedo)
        return spectral_albedo

    @property
    def broadband_albedo(self):
        """Calculate and return the broadband albedo"""
        broadband_albedo = np.trapz(
            self.spectral_albedo,
            self.spectral_albedo.coords["wavelength"],
        ) / (
            self.spectral_albedo.coords["wavelength"].max().values
            - self.spectral_albedo.coords["wavelength"].min().values
        )
        broadband_albedo = xr.DataArray(broadband_albedo, dims=["sample", "point"], name="broadband_albedo")
        return broadband_albedo

    def _gen_wavelength_point_df(self, raw_array: ndarray) -> pd.DataFrame:
        """Generate a dataframe with the wavelenghts as index and points as colums"""
        sample_df = pd.DataFrame(raw_array).transpose()
        sample_df["wavelenght (nm)"] = self.wavelengths
        sample_df.set_index("wavelenght (nm)", inplace=True)
        return sample_df
