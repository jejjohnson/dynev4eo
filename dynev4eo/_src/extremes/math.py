import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd





def calculate_sigma(threshold, location, scale, shape):
    """
    Calculates the sigma value using the given threshold, location, scale, and shape parameters.

    Parameters:
    threshold (float): The threshold value.
    location (float): The location parameter.
    scale (float): The scale parameter.
    shape (float): The shape parameter.

    Returns:
    float: The calculated sigma value.
    """
    return scale + shape * (threshold - location)
