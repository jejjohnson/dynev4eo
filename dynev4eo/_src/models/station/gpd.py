from typing import Optional
import equinox as eqx
from jaxtyping import Float, Array
from tensorflow_probability.substrates.jax import distributions as tfd
import jax
import xarray as xr
import jax.numpy as jnp
from scipy.stats import kurtosis
import numpyro.distributions as dist
import numpyro
from dynev4eo._src.extremes.math import calculate_sigma
from dynev4eo._src.extremes.returns import calculate_exceedence_probs, calculate_extremes_rate_pot, estimate_return_level_gpd, calculate_rate
from loguru import logger
import pandas as pd


class ProbGPDIID(eqx.Module):
    num_time_steps: int
    sigma: dist.Distribution
    concentration: dist.Distribution
    threshold: Array
    extremes_rate: Array
    name: str | None = None

    @classmethod
    def init_from_data(
        cls,
        da: xr.DataArray,
        extremes_rate=None,
        threshold=None,
        location=None,
        scale=None,
        shape=None,
        verbose: bool=True,
        ):
        
        y = da.dropna(dim="time")
        num_steps = y.shape[0]
        # calculate initial statistics
        if extremes_rate is None:
            num_timesteps = (da.time.values.max() - da.time.values.min())
            return_period_size = pd.to_timedelta("365.2524D")
            extremes_rate = jnp.asarray(calculate_extremes_rate_pot(y, num_timesteps, return_period_size))
        if location is None:
            location = y.mean().values
        if scale is None:
            scale = y.std().values
        if shape is None:
            shape = jnp.asarray(-0.2)
        if threshold is None:
            threshold = da.threshold.values
        if verbose:
            logger.info(f"Location: {location:.2f} | {location.shape}")
            logger.info(f"Scale: {scale:.2f} | {scale.shape}")
            logger.info(f"Kurtosis: {shape:.2f} | {shape.shape}")
            logger.info(f"Threshold: {threshold:.2f} | {threshold.shape}")
            logger.info(f"Extremes Rate: {extremes_rate:.2f} | {extremes_rate.shape}")
        
        # initialize distributions
        # scale = dist.LogNormal(jnp.asarray(scale), 0.25)
        sigma = dist.Uniform(0.0, 10.0)
        # concentration = dist.TruncatedNormal(-0.3, 0.1, low=-0.5, high=0.5)
        concentration = dist.Uniform(low=-0.5, high=0.0)
        
        return cls(
            num_time_steps=num_steps,
            name = da.name,
            sigma=sigma, 
            concentration=concentration,
            threshold=jnp.asarray(threshold),
            extremes_rate=jnp.asarray(extremes_rate)
        )

    def model(self, y: Float[Array, "T"]=None,  mask: Float[Array, "T"]=None):
        
        if y is not None:
            num_time_steps = y.shape[0]
        else:
            num_time_steps = self.num_time_steps

        # GEVD Parameters
        concentration = numpyro.sample("concentration", fn=self.concentration)
        sigma = numpyro.sample("sigma", fn=self.sigma)

        # time steps
        with numpyro.plate("time", num_time_steps):
            obs_dist = tfd.GeneralizedPareto(self.threshold, sigma, concentration)
            if mask is not None:
                obs_dist = tfd.Masked(obs_dist, mask)
                
            out = numpyro.sample("obs", obs_dist, obs=y if y is not None else None)

        rl_100 = numpyro.deterministic("return_level_100", estimate_return_level_gpd(100, self.threshold, sigma, concentration, rate=self.extremes_rate))
        return out
    

class ProbGPDIIDReparam(eqx.Module):
    num_time_steps: int
    location: dist.Distribution
    scale: dist.Distribution
    concentration: dist.Distribution
    threshold: Array
    extremes_rate: Array

    @classmethod
    def init_from_data(cls, da, extremes_rate=None, threshold=None, location=None, scale=None, shape=None):
        
        y = da.dropna(dim="time")
        num_steps =y.shape[0]
        # calculate initial statistics
        if extremes_rate is None:
            num_timesteps = (da.time.values.max() - da.time.values.min())
            return_period_size = pd.to_timedelta("365.2524D")
            extremes_rate = jnp.asarray(calculate_extremes_rate_pot(y, num_timesteps, return_period_size))
        if location is None:
            location = y.mean().values
        if scale is None:
            scale = y.std().values
        if shape is None:
            shape = jnp.asarray(-0.2)
        if threshold is None:
            threshold = da.threshold.values
        logger.info(f"Location: {location:.2f} | {location.shape}")
        logger.info(f"Scale: {scale:.2f} | {scale.shape}")
        logger.info(f"Kurtosis: {shape:.2f} | {shape.shape}")
        logger.info(f"Threshold: {threshold:.2f} | {threshold.shape}")
        logger.info(f"Extremes Rate: {extremes_rate:.2f} | {extremes_rate.shape}")
        
        # initialize distributions
        # location = dist.TruncatedNormal(jnp.asarray(location), jnp.asarray(scale))
        location = dist.Uniform(0, 100)
        # scale = dist.LogNormal(jnp.asarray(scale), 0.25)
        scale = dist.Uniform(0.0, 10.0)
        # concentration = dist.TruncatedNormal(-0.3, 0.1, low=-0.5, high=0.5)
        concentration = dist.Uniform(low=-0.5, high=0.0)
        
        return cls(
            num_time_steps=num_steps, 
            location=location, 
            scale=scale, 
            concentration=concentration,
            threshold=jnp.asarray(threshold),
            extremes_rate=jnp.asarray(extremes_rate)
        )

    def model(self, y: Float[Array, "T"]=None,  mask: Float[Array, "T"]=None):
        
        if y is not None:
            num_time_steps = y.shape[0]
        else:
            num_time_steps = self.num_time_steps

        # GEVD Parameters
        location = numpyro.sample("location", fn=self.location)
        scale = numpyro.sample("scale", fn=self.scale)
        concentration = numpyro.sample("concentration", fn=self.concentration)
        
        # GPD parameters        
        sigma = calculate_sigma(threshold=self.threshold, location=location, scale=scale, shape=concentration)
        sigma = numpyro.deterministic("sigma", sigma)

        # time steps
        with numpyro.plate("time", num_time_steps):
            obs_dist = tfd.GeneralizedPareto(self.threshold, sigma, concentration)
            if mask is not None:
                obs_dist = tfd.Masked(obs_dist, mask)
                
            out = numpyro.sample("obs", obs_dist, obs=y if y is not None else None)

        rl_100 = numpyro.deterministic("return_level_100", estimate_return_level_gpd(100, self.threshold, sigma, concentration, rate=self.extremes_rate))
        return out
    
    
class ProbGPDIIDNoPool(eqx.Module):
    num_time_steps: int
    location: dist.Distribution
    scale: dist.Distribution
    concentration: dist.Distribution
    threshold: dist.Distribution
    extremes_rate: Array

    @classmethod
    def init_from_data(cls, da, extremes_rate=None, threshold=None, location=None, scale=None, shape=None):
        
        y = da.dropna(dim="time")
        num_steps = y.shape[0]
        # calculate initial statistics
        if extremes_rate is None:
            num_timesteps = (da.time.values.max() - da.time.values.min())
            return_period_size = pd.to_timedelta("365.2524D")
            extremes_rate = jnp.asarray(calculate_extremes_rate_pot(y, num_timesteps, return_period_size))
        if location is None:
            location = y.mean().values
        if scale is None:
            scale = y.std().values
        if shape is None:
            shape = jnp.asarray(-0.2)
        if threshold is None:
            threshold = da.threshold.values
        logger.info(f"Location: {location:.2f} | {location.shape}")
        logger.info(f"Scale: {scale:.2f} | {scale.shape}")
        logger.info(f"Kurtosis: {shape:.2f} | {shape.shape}")
        logger.info(f"Threshold: {threshold:.2f} | {threshold.shape}")
        logger.info(f"Extremes Rate: {extremes_rate:.2f} | {extremes_rate.shape}")
        
        # initialize distributions
        # location = dist.TruncatedNormal(jnp.asarray(location), jnp.asarray(scale))
        location = dist.Uniform(0, 100)
        # scale = dist.LogNormal(jnp.asarray(scale), 0.25)
        scale = dist.Uniform(0.0, 10.0)
        # concentration = dist.TruncatedNormal(-0.3, 0.1, low=-0.5, high=0.5)
        concentration = dist.Uniform(low=-0.5, high=0.0)
        threshold = dist.Normal(threshold,5.0)
        
        return cls(
            num_time_steps=num_steps, 
            location=location, 
            scale=scale, 
            concentration=concentration,
            threshold=threshold,
            extremes_rate=jnp.asarray(extremes_rate)
        )

    def model(self, y: Float[Array, "T"]=None,  mask: Float[Array, "T"]=None):
        
        if y is not None:
            num_time_steps = y.shape[0]
        else:
            num_time_steps = self.num_time_steps

        # GEVD Parameters
        
        
        location = numpyro.sample("location", fn=self.location)
        concentration = numpyro.sample("concentration", fn=self.concentration)
        scale = numpyro.sample("scale", fn=self.scale)
        
        

        # time steps
        with numpyro.plate("time", num_time_steps):
            
            threshold = numpyro.sample("threshold", fn=self.threshold)
            sigma = calculate_sigma(threshold=threshold, location=location, scale=scale, shape=concentration)
            sigma = numpyro.deterministic("sigma", sigma)
            
            
            # GPD parameters        
            obs_dist = tfd.GeneralizedPareto(threshold, sigma, concentration)
            if mask is not None:
                obs_dist = tfd.Masked(obs_dist, mask)
                
            out = numpyro.sample("obs", obs_dist, obs=y if y is not None else None)

        rl_100 = numpyro.deterministic("return_level_100", estimate_return_level_gpd(100, threshold, sigma, concentration, rate=self.extremes_rate))
        return out