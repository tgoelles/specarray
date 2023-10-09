"""Class for Specim hyperspectral Camera data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numpy import ndarray
from spectral.io.bilfile import BilFile

from .io.specim_folder import from_specim_folder


@dataclass
class SpecArray:
    """Class for Specim hyperspectral Camera data"""

    capture: xr.DataArray
    metadata: dict
    wavelengths: pd.Series

    black: xr.DataArray
    white: xr.DataArray

    _capture_spectral: BilFile  # spectral lib data, lazy loaded

    @classmethod
    def from_folder(cls, folder: Path):
        """Create a SpecArray from a folder"""
        capture, metadata, wavelengths, black, white, capture_spectral = from_specim_folder(folder)
        return cls(
            capture=capture,
            metadata=metadata,
            wavelengths=wavelengths,
            black=black,
            white=white,
            _capture_spectral=capture_spectral,
        )

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
    def has_black(self) -> bool:
        """Return True if the black aka DARKREF reference is available"""
        return self.black.shape != ()

    @property
    def has_white(self) -> bool:
        """Return True if the white aka WHITEREF reference is available"""
        return self.white.shape != ()

    @property
    def spectral_albedo(self) -> xr.DataArray:
        """Calculate and return the spectral albedo. The values are limited to 0 and 1."""
        if self.has_black and self.has_white:
            black_mean = self.black.mean(dim="sample")
            white_mean = self.white.mean(dim="sample")
            spectral_albedo = (self.capture - black_mean) / (white_mean - black_mean)
            spectral_albedo = xr.where(spectral_albedo < 0.0, 0.0, spectral_albedo)
            spectral_albedo = xr.where(spectral_albedo > 1.0, 1.0, spectral_albedo)
            spectral_albedo.name = "spectral albedo"
            return spectral_albedo
        else:
            raise ValueError("No black or white reference")

    @property
    def broadband_albedo(self) -> xr.DataArray:
        """Calculate and return the broadband albedo"""
        if self.has_black and self.has_white:
            broadband_albedo = np.trapz(
                self.spectral_albedo.transpose(
                    "sample",
                    "point",
                    "wavelength",
                ),
                self.spectral_albedo.coords["wavelength"],
            ) / (
                self.spectral_albedo.coords["wavelength"].max().values
                - self.spectral_albedo.coords["wavelength"].min().values
            )
            broadband_albedo = xr.DataArray(broadband_albedo, dims=["sample", "point"], name="broadband albedo")
            return broadband_albedo
        else:
            raise ValueError("No black or white reference")
