from typing import Union
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

def calculate_rate(threshold, location, scale, shape):
    """
    Calculates the rate using the threshold, location, scale, and shape parameters.

    Args:
        threshold (float): The threshold value.
        location (float): The location parameter.
        scale (float): The scale parameter.
        shape (float): The shape parameter.

    Returns:
        float: The calculated rate.

    """
    z = (threshold - location) / scale
    z_shape_1 = 1 + shape * z 
    shape_minus_1 = - jnp.reciprocal(shape)
    return jnp.power(z_shape_1, shape_minus_1)


def calculate_exceedence_probs(
    y: np.ndarray,
    alpha: float=0.0,
    beta: float=1.0,
):

    # number of values
    Ny = len(y)
    
    # rank values from first (1) to the last (Ny)
    ranks = Ny + 1 - rankdata(y, method="average")
    
    # calculate exceedence probabilities
    # P = (r - α) / (Ny + 1 - α - β)
    exceedence_probs = (ranks - alpha) / (Ny + 1 - alpha - beta)
    
    return exceedence_probs


def calculate_extremes_rate_pot(
    y: np.ndarray,
    num_timesteps: Union[str, pd.Timedelta],
    return_period_size: Union[str, pd.Timedelta] = "365.2524D",
):
    # number of values
    num_extremes = len(y)

    # sanitize inputs
    if isinstance(return_period_size, str):
        return_period_size = pd.to_timedelta(return_period_size)

    if isinstance(num_timesteps, str):
        num_timesteps = pd.to_timedelta(num_timesteps)


    # Calculate rate of extreme events as number of events per one return period
    num_periods = num_timesteps / return_period_size
    extremes_rate = num_extremes / num_periods
    
    return extremes_rate


def calculate_extremes_rate_bm(
    block_size: Union[str, pd.Timedelta] = "365.2524D",
    return_period_size: Union[str, pd.Timedelta] = "365.2524D",
):

    # sanitize inputs
    if isinstance(return_period_size, str):
        return_period_size = pd.to_timedelta(return_period_size)

    if isinstance(block_size, str):
        block_size = pd.to_timedelta(block_size)

    # Calculate rate of extreme events as number of events per one return period
    extremes_rate = return_period_size / block_size
    
    return extremes_rate


def estimate_return_level_gevd(period, location, scale, shape):
    """
    Estimate the return level using the Generalized Extreme Value Distribution.

    Parameters:
    - period (float): The return period in years.
    - location (float): The location parameter of the GEV distribution.
    - scale (float): The scale parameter of the GEV distribution.
    - shape (float): The shape parameter of the GEV distribution.

    Returns:
    - float: The estimated return level.

    """
    return_period = 1 - 1 / period
    return tfd.GeneralizedExtremeValue(location, scale, shape).quantile(return_period)


def estimate_ari_gevd(period, location, scale, shape):
    """
    Estimate the Annual Return Interval (ARI) using the Generalized Extreme Value Distribution (GEVD).

    Args:
        period (float): The return period in years.
        location (float): The location parameter of the GEVD.
        scale (float): The scale parameter of the GEVD.
        shape (float): The shape parameter of the GEVD.

    Returns:
        float: The estimated ARI.

    """
    return_period = jnp.exp(-1 / period)
    return tfd.GeneralizedExtremeValue(location, scale, shape).quantile(return_period)


def estimate_return_level_gpd(period, location, scale, shape, rate=1.0):
    """
    Estimate the return level using the Generalized Pareto Distribution (GPD).

    Parameters:
    - period (float): The return period in years.
    - location (float): The location parameter of the GPD.
    - scale (float): The scale parameter of the GPD.
    - shape (float): The shape parameter of the GPD.
    - rate (float, optional): The rate parameter of the GPD. Default is 1.0.

    Returns:
    - float: The estimated return level.

    """
    return_period = 1 - 1 / (rate * period)
    return tfd.GeneralizedPareto(location, scale, shape).quantile(return_period)


def estimate_ari_gpd(period, location, scale, shape, rate):
    """
    Estimate the value at risk (VaR) using the Generalized Pareto Distribution (GPD).

    Args:
        period (float): The time period for which VaR is estimated.
        location (float): The location parameter of the GPD.
        scale (float): The scale parameter of the GPD.
        shape (float): The shape parameter of the GPD.
        rate (float): The rate parameter of the GPD.

    Returns:
        float: The estimated VaR using GPD.

    """
    return_period = jnp.exp(-1 / (rate * period))
    return tfd.GeneralizedPareto(location, scale, shape).quantile(return_period)
