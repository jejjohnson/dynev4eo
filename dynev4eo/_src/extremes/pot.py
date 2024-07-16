import xarray as xr
import numpy as np


def calculate_pot_quantile(da: xr.DataArray, quantile: float=0.98) -> float:
    """
    Calculate the quantile value for a given DataArray.

    Parameters:
        da (xr.DataArray): The input DataArray.
        quantile (float, optional): The quantile value to calculate (default is 0.98).

    Returns:
        float: The quantile value.

    """
    return da.quantile(q=[quantile], dim="time").values


def calculate_pot_extremes(ds_station, num_days: int=3, freq: str="D", quantile: float=0.95):
    """
    Load potential data from a dataset.

    Parameters:
    - ds_station: Dataset
        The dataset containing the station data.
    - num_days: int, optional
        The number of days for temporal discretization. Default is 3.
    - freq: str, optional
        The frequency for temporal discretization. Default is "D" (daily).
    - quantile: float, optional
        The quantile value used to calculate the threshold. Default is 0.95.

    Returns:
    - ds_magnitude: Dataset
        The dataset containing the potential data.

    """
    threshold_init = np.floor(calculate_pot_quantile(ds_station.t2m_max, quantile=quantile))
    ds_pot = ds_station.where(ds_station.t2m_max >= threshold_init, drop=False).sel(time=slice(None, "2022")).squeeze()

    # temporal discretization
    dt = f"{num_days}{freq}"
    # dt = "3D" # 3 Days # 1 Month
    ds_magnitude = ds_pot.resample(time=dt).max().fillna(np.nan)
    ds_magnitude["mask"] = np.isfinite(ds_magnitude.t2m_max)
    ds_magnitude["threshold"] = threshold_init
    ds_magnitude["num_days"] = num_days
    ds_magnitude["freq"] = freq 
    return ds_magnitude
