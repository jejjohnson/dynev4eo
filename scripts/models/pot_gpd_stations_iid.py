import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # first gpu
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'FALSE'

import numpyro
numpyro.set_platform("cpu")
import jax
jax.config.update('jax_platform_name', 'cpu')

import autoroot
import typer
import pandas as pd
import numpy as np
import xarray as xr
import arviz as az
from tqdm import tqdm
import jax.numpy as jnp
import jax.random as jrandom
from loguru import logger
import pint_xarray
import tqdm
from dynev4eo._src.io import MyPaths
from dynev4eo._src.preprocess.masks import add_country_mask
from dynev4eo._src.preprocess.validation import validate_longitude, validate_latitude
from dynev4eo._src.viz.ts import plot_ts_station_pot, plot_scatter_station, plot_histogram_station
from dynev4eo._src.inference import SVILearner, MCMCLearner
from dynev4eo._src.models.station.gevd import ProbGEVDIID
from dynev4eo._src.viz.params import plot_posterior_param_trace, plot_posterior_params_joint
from dynev4eo._src.viz.posterior import plot_posterior_predictive
from dynev4eo._src.viz.metrics import plot_qq_plot_gevd
from dynev4eo._src.viz.returns import plot_return_level_gevd, plot_return_level_hist
import numpyro
from numpyro.infer import Predictive
from dynev4eo._src.io import MyPaths, MySavePaths






def plot_histogram_density_station(ds_pot, figures_save_dir, savefilename):
    fig, ax = plt.subplots()
    ds_pot.t2m_max.plot.hist(ax=ax, bins=20, linewidth=2, density=True, fill=False)
    sns.kdeplot(
        np.asarray(ds_pot.t2m_max.values.ravel()), 
        color=f"black",
        linewidth=5, label="KDE Fit"
    )
    ax.set(
        ylabel="Density of Observations",
        title=f"{str(ds_pot.station_name.values).upper()} | Block Maximum",
    )
    plt.tight_layout()
    save_name = Path(figures_save_dir).joinpath(f"{savefilename}").joinpath("histogram")
    save_name.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_name.joinpath(f"{ds_pot.station_id.values}_t2mmax_pot_hist_density.png"))
    plt.close()


     

def calculate_posterior(mcmc, model, rng_key):
    # get posterior samples
    posterior_samples = mcmc.get_samples()

    # Get Posterior predictive Samples
    posterior_predictive = Predictive(
        model=model.model, posterior_samples=posterior_samples
    )
    # Posterior predictive samples
    rng_key, rng_subkey = jrandom.split(rng_key)
    posterior_predictive_samples = posterior_predictive(rng_subkey)

    # get posterior samples
    posterior_samples = mcmc.get_samples()

    # Get Posterior predictive Samples
    posterior_predictive = Predictive(
        model=model.model, posterior_samples=posterior_samples,
        return_sites=["location", "scale", "concentration", "return_level_100", "sigma", "obs"]
    )

    # Posterior predictive samples
    rng_key, rng_subkey = jrandom.split(rng_key)
    posterior_predictive_samples = posterior_predictive(rng_subkey)

    arxiv_summary = az.from_numpyro(
        posterior=mcmc,
        posterior_predictive=posterior_predictive_samples
    )

    # correct some coordinates
    arxiv_summary = arxiv_summary.rename({"obs_dim_0": "time"})

    return arxiv_summary


def calculate_return_levels(arxiv_summary, model):

    from bayesevt._src.models.gpd import estimate_return_level_gpd
    return_periods = np.logspace(-1.0, 3, 100)

    arxiv_summary = arxiv_summary.assign_coords({"return_period": return_periods})

    fn = jax.jit(estimate_return_level_gpd)

    def my_fn(location, scale, shape):
        # print(location.shape, scale.shape, shape.shape)
        rl = jax.vmap(fn, in_axes=(0,None,None,None, None))(return_periods, location, scale, shape, model.extremes_rate)
        return rl
    
    arxiv_summary.posterior_predictive["return_level"] = xr.apply_ufunc(
        my_fn,
        model.threshold,
        arxiv_summary.posterior_predictive.sigma,
        arxiv_summary.posterior_predictive.concentration,
        input_core_dims=None,
        output_core_dims=[["return_period"]],
        vectorize=True
    )

    arxiv_summary = arxiv_summary.assign_coords({"return_period": return_periods})

    return_period_empirical = 1 / calculate_exceedence_probs(arxiv_summary.observed_data.obs)
    return_period_empirical /= model.extremes_rate
    arxiv_summary.posterior_predictive["return_level_empirical"] = (("time"), return_period_empirical)

    return arxiv_summary


def plot_posterior_param_trace(arxiv_summary, figures_save_dir, savefilename):

    az.plot_trace(
        arxiv_summary, 
        var_names=["location", "scale", "sigma", "concentration", "return_level_100"],);
    plt.tight_layout()
    save_name = Path(figures_save_dir).joinpath(f"{savefilename}").joinpath("metrics")
    save_name.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_name.joinpath(f"{savefilename}_param_divergences.png")) 
    plt.tight_layout()
    plt.close()


def plot_posterior_predictive(arxiv_summary, figures_save_dir, savefilename):
    fig, ax = plt.subplots()
    az.plot_ppc(arxiv_summary, ax=ax, group="posterior", colors=["tab:blue", "tab:red", "black"])
    ax.set(
        xlabel="Temperature [°C]",
        # ylabel="Density",
    )
    plt.tight_layout()
    save_name = Path(figures_save_dir).joinpath(f"{savefilename}").joinpath("metrics")
    save_name.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_name.joinpath(f"{savefilename}_pred_prob_posterior.png")) 
    plt.close()


def plot_posterior_params_joint(arxiv_summary, figures_save_dir, savefilename):
    ax = az.plot_pair(
        arxiv_summary,
        group="posterior",
        var_names=["location", "scale", "sigma", "concentration", "return_level_100"],
        kind=["scatter", "kde"],
        kde_kwargs={"fill_last": False},
        marginals=True,
        # coords=coords,
        point_estimate="median",
        figsize=(11.5, 10),
    )
    save_name = Path(figures_save_dir).joinpath(f"{savefilename}").joinpath("params")
    save_name.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_name.joinpath(f"{savefilename}_params_prob_posterior.png")) 
    plt.close()


def plot_return_level(arxiv_summary, figures_save_dir, savefilename):


    fig, ax = plt.subplots(figsize=(6.5, 6))

    ciu, mu, cil = arxiv_summary.posterior_predictive["return_level"].quantile(q=[0.025, 0.5, 0.975], dim=["chain", "draw"])

    ax.plot(
        arxiv_summary.posterior_predictive.return_period.values, mu, 
        linestyle="-", linewidth=3, color="tab:red",
        label="GEVD (Mean)"
    )

    ax.fill_between(
        arxiv_summary.posterior_predictive.return_period.values, 
        cil, ciu,
        alpha=0.3,
        linestyle="solid",
        color="tab:red",
        label="95% Confidence Interval",
    )
    ax.set(
        xlabel="Return Period [Years]",
        ylabel="Return Levels [°C]",
        xscale="log",
    )

    ax.scatter(
        x=arxiv_summary.posterior_predictive.return_level_empirical,
        y=arxiv_summary.observed_data.obs,
        s=10.0, zorder=3, color="black",
        marker="o",
        label="Empirical Distribution",
    )


    # SECOND AXIS
    def safe_reciprocal(x):
        """Vectorized 1/x, treating x==0 manually"""
        x = np.array(x, float)
        near_zero = np.isclose(x, 0)
        x[near_zero] = np.inf
        x[~near_zero] = np.reciprocal(x[~near_zero])
        return x

    secax = ax.secondary_xaxis("top", functions=(safe_reciprocal, safe_reciprocal))
    secax.set_xlabel("Probability")
    secax.set_xticks([1.0, 0.1, 0.01, 0.001])
    # ax.set_xticks([1, 10, 100, 1000, 10000])

    # format log scale labels
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    secax.xaxis.set_major_formatter(formatter)

    plt.grid(which="both", visible=True)
    plt.legend()
    plt.tight_layout()
    save_name = Path(figures_save_dir).joinpath(f"{savefilename}").joinpath("returns")
    save_name.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_name.joinpath(f"{savefilename}_returns_prob_posterior_vs_empirical.png")) 
    plt.close()


def plot_return_level_hist(arxiv_summary, figures_save_dir, savefilename):
    rl_median = arxiv_summary.posterior_predictive.return_level_100.median()
    rl_mean = arxiv_summary.posterior_predictive.return_level_100.mean()

    fig, ax = plt.subplots()
    arxiv_summary.posterior_predictive.return_level_100.plot.hist(
        ax=ax,
        density=True, bins=20, linewidth=2, 
        fill=False, label="Histogram", zorder=3
    )
    sns.kdeplot(
        np.asarray(arxiv_summary.posterior_predictive.return_level_100.values.ravel()), 
        color=f"black",
        linewidth=5, label="KDE Fit"
    )
    ax.vlines(rl_median, ymin=-0.1, ymax=0.1, linewidth=5, color="tab:blue", zorder=4, label=f"Median: {rl_median:.2f}")
    ax.vlines(rl_mean, ymin=-0.1, ymax=0.1, linewidth=5, color="tab:orange", zorder=4, label=f"Mean: {rl_mean:.2f}")
    ax.set(
        xlabel="Return Levels [°C]",
        ylabel="Probability Density Function",
        title="Return Period @ 100 Years",
        ylim=[0.0, None]
    )
    plt.legend()
    plt.tight_layout()
    save_name = Path(figures_save_dir).joinpath(f"{savefilename}").joinpath("returns")
    save_name.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_name.joinpath(f"{savefilename}_returns_100yr_hist.png")) 
    plt.close()


def plot_qq_plot(arxiv_summary, figures_save_dir, savefilename):
    y_min = arxiv_summary.observed_data.obs.min().values
    y_max = arxiv_summary.observed_data.obs.max().values

    diff = 0.10 * (y_max - y_min)
    y_domain = np.linspace(y_min - diff, y_max + diff, 100)


    fn_quantile = lambda x: tfd.GeneralizedPareto(
        loc=jnp.asarray(arxiv_summary.posterior["threshold"].squeeze()),
        scale=jnp.asarray(arxiv_summary.posterior["sigma"].squeeze()),
        concentration=jnp.asarray(arxiv_summary.posterior["concentration"].squeeze()),
    ).quantile(x)

    xp = np.linspace(0, 1, 100)
    # calculate model quantiles
    model_quantile = jax.vmap(fn_quantile)(xp)
    # calculate empirical quantiles
    empirical_quantile = np.quantile(arxiv_summary.observed_data.obs.values, xp)

    mq_cl, mq_mu, mq_cu = np.quantile(model_quantile, q=[0.025, 0.5, 0.975], axis=1)

    fig, ax = plt.subplots()
    ax.scatter(mq_mu, empirical_quantile, color="tab:blue", s=30.0, zorder=3)
    ax.scatter(mq_cu, empirical_quantile, color="gray", s=10.0, alpha=0.4, zorder=3)

    ax.scatter(mq_cl, empirical_quantile, color="gray", s=10.0, alpha=0.4, zorder=3)
    ax.plot(y_domain, y_domain, color="black")

    ax.set(
        xlim=[y_domain[0], y_domain[-1]],
        ylim=[y_domain[0], y_domain[-1]],
        xlabel="Model Quantiles",
        ylabel="Empirical Quantiles"
        
    )
    ax.set_aspect('equal', 'box')
    ax.grid(which="both", visible=True)
    plt.tight_layout()
    save_name = Path(figures_save_dir).joinpath(f"{savefilename}").joinpath("metrics")
    save_name.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_name.joinpath(f"{savefilename}_qqplot.png")) 
    plt.close()
   

def main(quantile: str="95", num_days: int=3, freq: str="D"):
    rng_key = jrandom.PRNGKey(123)
    logger.info("Applying Links...")
    data_url = "/pool/usuarios/juanjohn/data/bayesevt/clean/observations/stations_spain.nc"
    figures_save_dir_core = Path("/pool/usuarios/juanjohn/projects/bayesevt/results/spain")
    inference = "mcmc"
    extreme_method = "pot"
    model_dist = "gpd"
    savefilename = f"{extreme_method}_{model_dist}_{inference}_q{quantile}_d{num_days}{freq}"

    logger.debug(f"Save File path: {figures_save_dir_core}")
    logger.debug(f"Save File Name: {savefilename}")

    logger.info("Opening dataset...")

    ds = xr.open_mfdataset(data_url).load()
    ds = add_country_mask(ds, "Spain") 
    ds = ds.where(ds.spain_mask==1, drop=True)#.t2m_max

    logger.info("Gathering stations...")

    station_ids = ds.station_id.values

    logger.debug(f"Total # Stations: {len(station_ids)}")

    pbar = tqdm.tqdm(station_ids)

    logger.info("Starting Loop...")
    for istation_id in pbar:
        

        pbar.set_description(f"Station ID: {istation_id}")

        ds_station = ds.where(ds.station_id == istation_id, drop=True).squeeze()

        figures_save_dir = figures_save_dir_core.joinpath(str(ds_station.station_id.values.squeeze()))

        # peak-over-threshold
        pbar.set_description(f"Station ID: {istation_id} | Calculating POT...")
        quantile_num = float(f"{0}.{quantile}")
        logger.debug(f"Quantile: {quantile_num}")
        ds_magnitude = load_pot_data(ds_station, num_days=num_days, freq=freq, quantile=quantile_num)


        pbar.set_description(f"Station ID: {istation_id} | Plotting EDA...")
        plot_ts_station(ds_station, ds_magnitude, figures_save_dir, savefilename)
        plot_scatter_station(ds_magnitude, figures_save_dir, savefilename)
        plot_histogram_station(ds_magnitude, figures_save_dir, savefilename)
        plot_histogram_density_station(ds_magnitude, figures_save_dir, savefilename)

        pbar.set_description(f"Station ID: {istation_id} | Calculating Model...")

        # initialize GEVD Model
        pbar.set_description(f"Station ID: {istation_id} | Initializing MAP Model...")
        y = ds_magnitude.t2m_max.dropna(dim="time")
        model = GPDTemporal.init_from_data(
            ds_magnitude.t2m_max, 
            threshold=np.asarray(ds_magnitude.threshold.values.item())
            )


        # get prior predictions
        pbar.set_description(f"Station ID: {istation_id} | Calculating MAP Results...")
        svi_results, params = fit_map(model, y, rng_key)

        pbar.set_description(f"Station ID: {istation_id} | Calculating MCMC Results...")
        mcmc = fit_mcmc(model, y, rng_key, params=params)

        # correct coordinates
        pbar.set_description(f"Station ID: {istation_id} | Creating Posterior DS...")
        arxiv_summary = calculate_posterior(mcmc, model, rng_key)
        arxiv_summary = arxiv_summary.assign_coords({"time": ds_magnitude.dropna(dim="time").time})

        # calculate return period
        pbar.set_description(f"Station ID: {istation_id} |Calculating return level...")
        arxiv_summary = calculate_return_levels(arxiv_summary, model)

        pbar.set_description(f"Station ID: {istation_id} | Adding MAP params...")
        arxiv_summary.posterior["threshold"] = ds_magnitude.threshold.squeeze()
        arxiv_summary.posterior["extremes_rate"] = np.asarray(model.extremes_rate)
        arxiv_summary.posterior["map_concentration"] = params["concentration"].squeeze()
        arxiv_summary.posterior["map_scale"] = params["scale"].squeeze()
        arxiv_summary.posterior["map_location"] = params["location"].squeeze()
        
        # adding statistics
        pbar.set_description(f"Station ID: {istation_id} | Adding Metrics...")
        out = az.waic(arxiv_summary, )
        arxiv_summary.log_likelihood["elpd_waic"] = out.elpd_waic
        arxiv_summary.log_likelihood["elpd_waic_se"] = out.se
        arxiv_summary.log_likelihood["p_waic"] = out.p_waic

        pbar.set_description(f"Station ID: {istation_id} | Saving DS...")
        save_name = Path(figures_save_dir).joinpath(f"{savefilename}").joinpath("results")
        save_name.mkdir(parents=True, exist_ok=True)
        arxiv_summary.to_netcdf(str(save_name.joinpath("results.nc")), engine="netcdf4")
        
        # PLOTS
        pbar.set_description(f"Station ID: {istation_id} | Plotting Parameter Traces...")
        plot_posterior_param_trace(arxiv_summary, figures_save_dir, savefilename)
        plot_posterior_predictive(arxiv_summary, figures_save_dir, savefilename)
        plot_posterior_params_joint(arxiv_summary, figures_save_dir, savefilename)
        plot_qq_plot(arxiv_summary, figures_save_dir, savefilename) 

        # calculate return period
        pbar.set_description(f"Station ID: {istation_id} | Plot Return Level...")
        plot_return_level(arxiv_summary=arxiv_summary, figures_save_dir=figures_save_dir, savefilename=savefilename)
        plot_return_level_hist(arxiv_summary=arxiv_summary, figures_save_dir=figures_save_dir, savefilename=savefilename)

        


        # del model, mcmc, svi_results

    
    pass
            

if __name__ == '__main__':
    typer.run(main)