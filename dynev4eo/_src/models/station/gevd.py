import equinox as eqx
from jaxtyping import Float, Array
from tensorflow_probability.substrates.jax import distributions as tfd
import jax.numpy as jnp
from scipy.stats import kurtosis
import numpyro.distributions as dist
import numpyro
from dynev4eo._src.extremes.returns import estimate_return_level_gevd, calculate_rate
from dynev4eo._src.extremes.math import calculate_sigma

from loguru import logger


class ProbGEVDIID(eqx.Module):
    num_time_steps: int
    location: dist.Distribution
    scale: dist.Distribution
    concentration: dist.Distribution
    return_periods: Array
    threshold: Array

    @classmethod
    def init_from_data(cls, y: Float[Array, "T"], location=None, scale=None, shape=None, threshold=None):
        num_steps = y.shape[0]
        # calculate initial statistics
        if location is None:
            location = jnp.mean(y)
        if scale is None:
            scale = jnp.std(y)
        if shape is None:
            shape = 0.1 * kurtosis(y)
        if threshold is None:
            threshold = y.min()
        shape = shape * jnp.ones_like(location)
        logger.info(f"Location: {location:.2f}")
        logger.info(f"Scale: {scale:.2f}")
        logger.info(f"Kurtosis: {shape:.2f}")
        logger.info(f"Threshold: {threshold:.2f}")
        # initialize distributions
        # location = dist.Normal(location, scale)
        location = dist.Uniform(location - 10, location + 10)
        # scale = dist.HalfNormal(scale * 0.5)
        # scale = dist.LogNormal(scale, 0.25)
        scale = dist.Uniform(0.0, 10.0)
        # concentration = dist.TruncatedNormal(-0.3, 0.05, low=-0.5, high=0.5)
        concentration = dist.Uniform(low=-0.5, high=0.0)
        
        return cls(
            num_time_steps=num_steps, 
            location=location, 
            scale=scale, 
            concentration=concentration,
            threshold=threshold,
            return_periods=jnp.logspace(0.001, 3, 100),
        )

    def model(self, y: Float[Array, "T"]=None):
        if y is not None:
            num_time_steps = y.shape[0]
        else:
            num_time_steps = self.num_time_steps
        loc = numpyro.sample("location", fn=self.location)
        scale = numpyro.sample("scale", fn=self.scale)
        concentration = numpyro.sample("concentration", fn=self.concentration)

        rate = numpyro.deterministic("rate", calculate_rate(location=loc, scale=scale, shape=concentration, threshold=self.threshold))
        sigma = numpyro.deterministic("sigma", calculate_sigma(threshold=self.threshold, location=loc, scale=scale, shape=concentration))

        
        # time trend
        with numpyro.plate("time", num_time_steps):
            out = numpyro.sample("obs", tfd.GeneralizedExtremeValue(loc, scale, concentration), obs=y if y is not None else None)
            
        rl = numpyro.deterministic("return_level", estimate_return_level_gevd(self.return_periods, loc, scale, concentration))

        rl_100 = numpyro.deterministic("return_level_100", estimate_return_level_gevd(100, loc, scale, concentration))
        return out