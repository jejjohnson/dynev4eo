from typing import Dict, Callable, List, Optional
import numpyro
from numpyro.infer.autoguide import AutoDelta, AutoLaplaceApproximation, AutoGuide
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer import init_to_value
from dataclasses import dataclass
from jaxtyping import PyTree, Array, PRNGKeyArray
import equinox as eqx
import arviz as az


class SVILearner(eqx.Module):
    """
    Stochastic Variational Inference (SVI) Learner.

    This class represents a learner for performing stochastic variational inference.
    It takes a probabilistic model, a guide, and other parameters to perform SVI.

    Args:
        model (PyTree): The probabilistic model.
        method (str, optional): The method to use for the guide initialization. 
            Defaults to "delta".
        num_steps (int, optional): The number of optimization steps. Defaults to 10,000.
        step_size (float, optional): The step size for the optimizer. Defaults to 1e-3.
        clip_norm (float, optional): The norm value to clip gradients. Defaults to 0.1.
        num_samples (int, optional): The number of samples to estimate the ELBO. Defaults to 10.

    Attributes:
        model (PyTree): The probabilistic model.
        guide (AutoGuide): The guide for the model.
        optimizer (Callable): The optimizer for the SVI.
        svi (PyTree): The SVI object.
        num_steps (int): The number of optimization steps.
        step_size (float): The step size for the optimizer.
        clip_norm (float): The norm value to clip gradients.
        num_samples (int): The number of samples to estimate the ELBO.
    """

    model: PyTree
    guide: AutoGuide
    optimizer: Callable
    svi: PyTree 
    num_steps: int = 10_000
    step_size: float = 1e-3
    clip_norm: float = 0.1
    num_samples: int = 10
    
    def __init__(
            self,
            model: PyTree,
            method: str="delta",
            num_steps: int = 10_000,
            step_size: float = 1e-3,
            clip_norm: float = 0.1,
            num_samples: int = 10,
        ):
        """
        Initializes the SVILearner.

        Args:
            model (PyTree): The probabilistic model.
            method (str, optional): The method to use for the guide initialization. 
                Defaults to "delta".
            num_steps (int, optional): The number of optimization steps. Defaults to 10,000.
            step_size (float, optional): The step size for the optimizer. Defaults to 1e-3.
            clip_norm (float, optional): The norm value to clip gradients. Defaults to 0.1.
            num_samples (int, optional): The number of samples to estimate the ELBO. Defaults to 10.
        """

        self.model = model
        self.clip_norm = clip_norm
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.step_size = step_size
        # initialize optimizer
        self.optimizer = numpyro.optim.ClippedAdam(step_size=self.step_size, clip_norm=self.clip_norm)

        # initialize guide
        if method.lower() in ["delta", "map"]:
            self.guide = AutoDelta(self.model)
        elif method.lower() == "laplace":
            self.guide = AutoLaplaceApproximation(self.model)
        else:
            raise ValueError(f"Unrecognized method: {method}")

        # initialize ELBO
        self.svi = SVI(self.model, self.guide, self.optimizer, loss=Trace_ELBO(self.num_samples))

    def __call__(self, rng_key, **kwargs):
        """
        Runs the SVI and returns the SVIPosterior.

        Args:
            rng_key (jax.random.PRNGKey): The random number generator key.
            **kwargs: Additional keyword arguments to pass to the SVI run method.

        Returns:
            SVIPosterior: The posterior object obtained from the SVI.
        """
        # run svi
        svi_result = self.svi.run(rng_key, num_steps=self.num_steps, **kwargs)
        
        return SVIPosterior(model=self.model, guide=self.guide, svi_result=svi_result)
    

class SVIPosterior(eqx.Module):
    """
    Represents a stochastic variational inference posterior.

    Args:
        model (Callable): The model function.
        guide (PyTree): The guide function.
        svi_result (PyTree): The result of stochastic variational inference.

    Attributes:
        params: The parameters of the posterior.
        median_params: The median parameters of the posterior.
        quantile_params: The quantile parameters of the posterior.
    """

    model: Callable
    guide: PyTree
    svi_result: PyTree

    @property
    def params(self):
        """
        Get the parameters of the posterior.

        Returns:
            PyTree: The parameters of the posterior.
        """
        return self.svi_result.params

    @property
    def median_params(self):
        """
        Get the median parameters of the posterior.

        Returns:
            PyTree: The median parameters of the posterior.
        """
        return self.guide.median(self.params)
    
    @property
    def quantile_params(self, quantiles: List[float]):
        """
        Get the quantile parameters of the posterior.

        Args:
            quantiles (List[float]): The quantiles to compute.

        Returns:
            PyTree: The quantile parameters of the posterior.
        """
        return self.guide.median(self.params)
    
    def variational_samples(self, rng_key: PRNGKeyArray, num_samples: int=100, **kwargs):
        """
        Generate samples from the variational posterior.

        Args:
            rng_key: The random number generator key.
            num_samples (int): The number of samples to generate.
            **kwargs: Additional keyword arguments.

        Returns:
            PyTree: The generated samples from the variational posterior.
        """
        samples = self.guide.sample_posterior(rng_key=rng_key, params=self.params, sample_shape=(num_samples,))
        return samples

    def posterior_samples(self, rng_key: PRNGKeyArray, num_samples: int=100, **kwargs):
        """
        Generate samples from the posterior.

        Args:
            rng_key: The random number generator key.
            num_samples (int): The number of samples to generate.
            **kwargs: Additional keyword arguments.

        Returns:
            PyTree: The generated samples from the posterior.
        """
        predictive = Predictive(model=self.guide, params=self.params, num_samples=num_samples)
        samples = predictive(rng_key=rng_key, **kwargs)
        return samples
    
    def posterior_predictive_samples(self, rng_key: PRNGKeyArray, num_samples: int=100, **kwargs):
        """
        Generate samples from the posterior predictive distribution.

        Args:
            rng_key: The random number generator key.
            num_samples (int): The number of samples to generate.
            **kwargs: Additional keyword arguments.

        Returns:
            PyTree: The generated samples from the posterior predictive distribution.
        """
        predictive = Predictive(model=self.model, guide=self.guide, params=self.params, num_samples=num_samples)
        samples = predictive(rng_key=rng_key, **kwargs)
        return samples


class MCMCLearner(eqx.Module):
    """
    MCMCLearner is a class that performs MCMC sampling using NUTS algorithm.

    Args:
        model (PyTree): The probabilistic model to perform inference on.
        target_accept_prob (float, optional): The target acceptance probability for the NUTS algorithm. Defaults to 0.99.
        num_warmup (int, optional): The number of warmup iterations for the MCMC sampling. Defaults to 2_000.
        num_samples (int, optional): The number of samples to draw from the posterior distribution. Defaults to 2_000.
        progress_bar (bool, optional): Whether to display a progress bar during sampling. Defaults to True.
        init_params (Optional[Dict[str, Array]], optional): Initial parameter values for the model. Defaults to None.
    """

    model: PyTree
    target_accept_prob: float = 0.99
    num_warmup: int = 2_000
    num_samples: int = 2_000
    progress_bar: bool = True
    init_params: Optional[Dict[str, Array]] = None

    def __call__(self, rng_key: PRNGKeyArray, **kwargs):
        """
        Perform MCMC sampling using NUTS algorithm.

        Args:
            rng_key (jax.random.PRNGKey): The random number generator key.
            **kwargs: Additional keyword arguments to be passed to the MCMC sampling.

        Returns:
            MCMCPosterior: The posterior distribution obtained from the MCMC sampling.
        """

        # initialize kernel
        nuts_kernel = NUTS(
            model=self.model, 
            target_accept_prob=0.99,
            init_strategy=init_to_value(values=self.init_params) if self.init_params is not None else None,
        )

        mcmc = MCMC(nuts_kernel, num_warmup=self.num_warmup, num_samples=self.num_samples, num_chains=1, progress_bar=self.progress_bar)

        mcmc.run(rng_key=rng_key, **kwargs)

        return MCMCPosterior(model=self.model, mcmc=mcmc)
    

class MCMCPosterior(eqx.Module):
    """
    Represents a class for performing MCMC inference on a given model.

    Attributes:
        model (PyTree): The model to perform inference on.
        mcmc (MCMC): The MCMC sampler used for inference.
    """

    model: PyTree
    mcmc: MCMC

    @property
    def posterior_samples(self):
        """
        Get the posterior samples generated by the MCMC sampler.

        Returns:
            The posterior samples.
        """
        return self.mcmc.get_samples()
    
    def posterior_predictive_samples(self, rng_key: PRNGKeyArray, **kwargs):
        """
        Generate posterior predictive samples using the MCMC sampler.

        Args:
            rng_key (jax.random.PRNGKey): The random number generator key.
            num_samples (int): The number of posterior predictive samples to generate.
            **kwargs: Additional keyword arguments to pass to the Predictive class.

        Returns:
            The posterior predictive samples.
        """

        posterior_samples = self.posterior_samples()

        posterior_predictive = Predictive(
            model=self.model, posterior_samples=posterior_samples
        )

        samples = posterior_predictive(rng_key=rng_key, **kwargs)

        return samples
    
    def init_arviz_summary(self):
        """
        Initialize an ArviZ summary object from the MCMC sampler.

        Returns:
            The ArviZ summary object.
        """
        return az.from_numpyro(self.mcmc)

    