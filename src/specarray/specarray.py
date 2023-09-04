from __future__ import annotations


import spectral.io.envi as envi
from spectral.io.bilfile import BilFile
from dataclasses import dataclass
from pathlib import Path
import xarray as xr
import pandas as pd
from numpy import ndarray

import dask.array as da
import numpy as np


@dataclass
class SpecArray:
    """Class for Specim hyperspectral Camera data"""

    folder: Path
    capture: xr.DataArray = None
    black: xr.DataArray = None
    white: xr.DataArray = None
    metadata: dict = None
    wavelengths: pd.Series = None

    def __post_init__(self):
        """Initialize the properties of the class"""
        self.spectral_data = self._load_spectral_data(mode="capture")
        self.metadata = self.spectral_data.metadata
        self.wavelengths = self._get_wavelenghts()

        modes = ["capture", "black", "white"]
        for mode in modes:
            try:
                spectral_data = self._load_spectral_data(mode=mode)
                data_array = self._create_data_array(spectral_data, mode)
                setattr(self, mode, data_array)
            except Exception as exception:
                print(f"An error occurred reading {mode}: {exception}")

    def __len__(self):
        """Return the number of records"""
        return len(self.capture)

    def __getitem__(self, key):
        return self.capture[key]

    def _create_data_array(self, spectral_data: BilFile, mode: str) -> xr.DataArray:
        data_array = xr.DataArray(
            da.from_array(spectral_data.asarray()),
            dims=["sample", "point", "band"],
            name=mode,
        )
        data_array = data_array.assign_coords(band=self.wavelengths.values)
        data_array = data_array.rename({"band": "wavelength"})
        return data_array

    @property
    def shape(self):
        """Return the shape of the data"""
        return self.capture.shape

    @property
    def spectral_albedo(self):
        spectral_albedo = (self.capture - self.black.mean(dim="sample")) / (
            self.white.mean(dim="sample") - self.black.mean(dim="sample")
        )
        spectral_albedo = xr.where(spectral_albedo < 0.0, 0.0, spectral_albedo)
        spectral_albedo = xr.where(spectral_albedo > 1.0, 1.0, spectral_albedo)
        return spectral_albedo

    @property
    def broadband_albedo(self):
        broadband_albedo = np.trapz(
            self.spectral_albedo,
            self.spectral_albedo.coords["wavelength"],
        ) / (
            self.spectral_albedo.coords["wavelength"].max().values
            - self.spectral_albedo.coords["wavelength"].min().values
        )
        broadband_albedo = xr.DataArray(broadband_albedo, dims=["sample", "point"], name="broadband_albedo")
        return broadband_albedo

    def _find_files(self):
        """Find all files in the folder"""
        capture_folder = self.folder / "capture"
        hdr_files = list(capture_folder.glob("*.hdr"))
        raw_files = list(capture_folder.glob("*.raw"))
        return hdr_files, raw_files

    def _load_spectral_data(self, mode="capture"):
        """Load the spectral data"""
        hdr_files, raw_files = self._find_files()

        hdr_file_path = None
        raw_file_path = None

        for hdr_file, raw_file in zip(hdr_files, raw_files):
            if mode == "capture":
                if not hdr_file.stem.startswith(("WHITEREF", "DARKREF")):
                    hdr_file_path = hdr_file
                    raw_file_path = raw_file
                    break
            elif mode == "white":
                prefix = "WHITEREF"
                if hdr_file.stem.startswith(prefix):
                    hdr_file_path = hdr_file
                    raw_file_path = raw_file
                    break
            elif mode == "black":
                prefix = "DARKREF"
                if hdr_file.stem.startswith(prefix):
                    hdr_file_path = hdr_file
                    raw_file_path = raw_file
                    break

        if hdr_file_path is None or raw_file_path is None:
            raise ValueError(f"No matching file found for mode {mode}")

        return envi.open(hdr_file_path, raw_file_path)

    def _get_wavelenghts(self):
        """Set the wavelenghts"""
        return pd.Series(pd.to_numeric(self.metadata["wavelength"]), name="wavelenghts (nm)")

    def _gen_wavelength_point_df(self, raw_array: ndarray):
        """Generate a dataframe with the wavelenghts as index and points as colums"""
        sample_df = pd.DataFrame(raw_array).transpose()
        sample_df["wavelenght (nm)"] = self.wavelengths
        sample_df.set_index("wavelenght (nm)", inplace=True)
        return sample_df
