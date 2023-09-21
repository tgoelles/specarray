from pathlib import Path

import dask.array as da
import pandas as pd
import spectral.io.envi as envi
import xarray as xr
from spectral.io.bilfile import BilFile

import warnings


def _find_files(folder: Path):
    """Find all files in the folder"""
    capture_folder = folder / "capture"
    hdr_files = list(capture_folder.glob("*.hdr"))
    raw_files = list(capture_folder.glob("*.raw"))
    return hdr_files, raw_files


def _load_spectral_data(folder: Path, mode="capture"):
    """Load the spectral data"""
    hdr_files, raw_files = _find_files(folder)

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


def _extract_wavelengths(metadata) -> pd.Series:
    """Return the wavelengths as a pandas series"""
    return pd.Series(pd.to_numeric(metadata["wavelength"]), name="wavelengths (nm)")


def _create_data_array(spectral_data: BilFile, mode: str, wavelengths: pd.Series) -> xr.DataArray:
    data_array = xr.DataArray(
        da.from_array(spectral_data.asarray()),
        dims=["sample", "point", "band"],
        name=mode,
    )
    data_array = data_array.assign_coords(band=wavelengths.values)
    data_array = data_array.rename({"band": "wavelength"})
    return data_array


def from_specim_folder(
    folder: Path,
) -> "tuple[xr.DataArray, dict, pd.Series, xr.DataArray, xr.DataArray]":
    """Create a SpecArray from a folder"""
    spectral_data = _load_spectral_data(folder=folder, mode="capture")
    metadata = spectral_data.metadata
    wavelengths = _extract_wavelengths(metadata=metadata)

    modes = ["capture", "black", "white"]
    black = xr.DataArray()
    white = xr.DataArray()
    capture = xr.DataArray()
    for mode in modes:
        try:
            spectral_data = _load_spectral_data(folder, mode)
            data_array = _create_data_array(spectral_data, mode, wavelengths)
            if mode == "capture":
                capture = data_array
                if capture.size == 0:
                    raise ValueError("No capture data found")
            elif mode == "black":
                black = data_array
                # warning that there is no Dark Reference
                warnings.warn("No Dark Reference found")
            elif mode == "white":
                white = data_array
                warnings.warn("No White Reference found")
        except Exception as exception:
            print(f"An error occurred reading {mode}: {exception}")
    return capture, metadata, wavelengths, black, white
