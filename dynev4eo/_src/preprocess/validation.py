import xarray as xr
import numpy as np


def transform_360_to_180(coord: np.ndarray) -> np.ndarray:
    """
    This function converts the coordinates that are bounded from [-180, 180]
    to coordinates bounded by [0, 360].

    Args:
        coord (np.ndarray): The input array of coordinates.

    Returns:
        np.ndarray: The output array of coordinates.
    """
    return (coord + 180) % 360 - 180


def transform_180_to_360(coord: np.ndarray) -> np.ndarray:
    """
    This function converts the coordinates that are bounded from [0, 360] to coordinates bounded by [-180, 180].

    Args:
        coord (np.ndarray): The input array of coordinates.

    Returns:
        np.ndarray: The output array of coordinates.
    """
    return coord % 360


def transform_180_to_90(coord: np.ndarray) -> np.ndarray:
    """
    This function converts the coordinates that are bounded from [-180, 180]
    to coordinates bounded by [0, 360].

    Args:
        coord (np.ndarray): The input array of coordinates.

    Returns:
        np.ndarray: The output array of coordinates.
    """
    return (coord + 90) % 180 - 90


def transform_90_to_180(coord: np.ndarray) -> np.ndarray:
    """
    This function converts the coordinates that are bounded from [0, 360]
    to coordinates bounded by [-180, 180].

    Args:
        coord (np.ndarray): The input array of coordinates.

    Returns:
        np.ndarray: The output array of coordinates.
    """
    return coord % 180


def validate_longitude(ds: xr.Dataset) -> xr.Dataset:
    """Format lat and lon variables

    Set units, ranges and names

    Args:
        ds: input data

    Returns:
        formatted data
    """
    new_ds = ds.copy()

    new_ds = _rename_longitude(new_ds)

    ds_attrs = new_ds.lon.attrs

    new_ds["lon"] = transform_360_to_180(new_ds.lon)
    new_ds["lon"] = new_ds.lon.assign_attrs(
        **{
            **ds_attrs,
            **dict(
                units="degrees_east",
                standard_name="longitude",
                long_name="Longitude",
            ),
            
        }
    )

    return new_ds


def validate_latitude(ds: xr.Dataset) -> xr.Dataset:

    new_ds = ds.copy()

    new_ds = _rename_latitude(new_ds)

    ds_attrs = new_ds.lat.attrs

    new_ds["lat"] = transform_180_to_90(new_ds.lat)
    new_ds["lat"] = new_ds.lat.assign_attrs(
        **{
            **ds_attrs,
            **dict(
                units="degrees_north",
                standard_name="latitude",
                long_name="Latitude",
            ),
        }
    )

    return new_ds


def _rename_longitude(ds):
    try:
        ds = ds.rename({"longitude": "lon"})
    except:
        pass
    return ds

def _rename_latitude(ds):
    try:
        ds = ds.rename({"latitude": "lat"})
    except:
        pass
    return ds