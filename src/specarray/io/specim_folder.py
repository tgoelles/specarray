from pathlib import Path
from os import listdir

import dask.array as da
import pandas as pd
import spectral.io.envi as envi
import xarray as xr
from spectral.io.bilfile import BilFile


import warnings


def _extract_wavelengths(metadata) -> pd.Series:
    """Return the wavelengths as a pandas series"""
    return pd.Series(pd.to_numeric(metadata["wavelength"]), name="wavelengths (nm)")


def _create_data_array(spectral_data: BilFile, mode: str, wavelengths: pd.Series) -> xr.DataArray:
    data_array = xr.DataArray(
        da.from_array(spectral_data._memmap, chunks=("auto", -1, -1)),
        dims=["sample", "band", "point"],
        name=mode,
    )
    data_array = data_array.assign_coords(band=wavelengths.values)
    data_array = data_array.rename({"band": "wavelength"})
    return data_array


def from_specim_folder(
    folder: Path,
) -> "tuple[xr.DataArray, dict, pd.Series, xr.DataArray, xr.DataArray, BilFile]":
    """Create a SpecArray from a folder"""

    modes = [
        "capture",
        "DARKREF_",
        "WHITEREF_",
    ]
    names = {"capture": "capture", "DARKREF_": "black", "WHITEREF_": "white"}
    black = xr.DataArray()
    white = xr.DataArray()
    capture = xr.DataArray()

    capture_folder = folder / "capture"

    for mode in modes:
        if mode == "capture":
            hdr_file_path = [
                capture_folder / file
                for file in listdir(capture_folder)
                if not file.startswith("WHITEREF") and not file.startswith("DARKREF") and file.endswith(".hdr")
            ]
            raw_file_path = [
                capture_folder / file
                for file in listdir(capture_folder)
                if not file.startswith("WHITEREF") and not file.startswith("DARKREF") and file.endswith(".raw")
            ]
        else:
            hdr_file_path = list(capture_folder.glob(f"{mode}*.hdr"))
            raw_file_path = list(capture_folder.glob(f"{mode}*.raw"))

        if len(hdr_file_path) > 0 and len(raw_file_path) > 0:
            spectral_data = envi.open(hdr_file_path[0], raw_file_path[0])
            metadata = spectral_data.metadata
            wavelengths = _extract_wavelengths(metadata=metadata)
            data_array = _create_data_array(spectral_data, mode, wavelengths)
            data_array.name = names[mode]
        else:
            break

        if mode == "capture":
            capture = data_array
            capture_spectral = spectral_data
            metadata = metadata
            wavelengths = wavelengths
        elif mode == "DARKREF_":
            black = data_array
        elif mode == "WHITEREF_":
            white = data_array

        if len(hdr_file_path) == 0:
            warnings.warn(f"No {mode} file found")

    if len(capture) == 0:
        raise ValueError("No capture file found")
    return capture, metadata, wavelengths, black, white, capture_spectral
