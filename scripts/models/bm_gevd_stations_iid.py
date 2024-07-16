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
from dynev4eo._src.viz.ts import plot_ts_station_bm, plot_scatter_station, plot_histogram_station
from dynev4eo._src.inference import SVILearner, MCMCLearner
from dynev4eo._src.models.station.gevd import ProbGEVDIID
from dynev4eo._src.viz.params import plot_posterior_param_trace, plot_posterior_params_joint
from dynev4eo._src.viz.posterior import plot_posterior_predictive
from dynev4eo._src.viz.metrics import plot_qq_plot_gevd
from dynev4eo._src.viz.returns import plot_return_level_gevd, plot_return_level_hist
import numpyro
from numpyro.infer import Predictive
from dynev4eo._src.io import MyPaths, MySavePaths

# initialize my paths
MY_ROOT_PATHS = MyPaths.init_from_dot_env()

# data URL
DATA_URL = MY_ROOT_PATHS.data_clean_dir.joinpath("t2m_stations_feten_spain.zarr")

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
        model=model.model, posterior_samples=posterior_samples
    )

    # Posterior predictive samples
    rng_key, rng_subkey = jrandom.split(rng_key)
    posterior_predictive_samples = posterior_predictive(rng_subkey)

    arxiv_summary = az.from_numpyro(
        posterior=mcmc,
        posterior_predictive=posterior_predictive_samples
    )

    # correct some coordinates
    arxiv_summary = arxiv_summary.rename({"return_level_dim_0": "return_period"})
    
    arxiv_summary = arxiv_summary.rename({"obs_dim_0": "time"})

    return arxiv_summary

app = typer.Typer()


@app.command()
def run_experiment(seed: int=123):

    rng_key = jrandom.PRNGKey(seed)

    logger.info(f"Starting script...")
    logger.debug(f"Save Path: {DATA_URL}")

    logger.info(f"Creating Method name")
    inference = "mcmc"
    extreme_method = "bm"
    model_dist = "gevd"
    stage = "experiments/iid"
    method_name = f"{extreme_method}_{model_dist}_{inference}"
    logger.info(f"Method Name: {method_name}")

    method_name = f"{extreme_method}_{model_dist}_{inference}"


    ds = xr.open_dataset(DATA_URL, engine="zarr")
    ds = ds.where(ds.spain_mask==1, drop=True)#.t2m_max


    logger.info("Gathering stations...")

    station_ids = ds.station_id.values

    logger.debug(f"Total # Stations: {len(station_ids)}")

    pbar = tqdm.tqdm(station_ids)


    logger.info("Starting Loop...")
    for istation_id in pbar:

        pbar.set_description(f"Station ID: {istation_id}")

        ds_station = ds.where(ds.station_id == istation_id, drop=True).squeeze()

        figures_save_paths = MySavePaths(
            base_path=MY_ROOT_PATHS.figures_dir,
            stage=stage,
            method=method_name,
            region=str(ds_station.station_id.values.squeeze())
            )
        
        figures_save_paths.make_dir()
        fig_save_path = figures_save_paths.full_path

        # block maximum
        pbar.set_description(f"Station ID: {istation_id} | Calculating BM...")
        ds_bm = ds_station.resample(time="1YE").max().sel(time=slice(None, "2022"))

        pbar.set_description(f"Station ID: {istation_id} | Plotting EDA...")
        plot_ts_station_bm(ds_station, ds_bm, fig_save_path)
        plot_scatter_station(ds_bm, fig_save_path)
        plot_histogram_station(ds_bm, fig_save_path)

        pbar.set_description(f"Initializing data and threshold...")
        y = jnp.asarray(ds_bm.t2m_max.values.squeeze())
        threshold = ds_station.t2m_max.quantile(q=0.95)

        pbar.set_description(f"Station ID: {istation_id} | Calculating Model...")
        model = ProbGEVDIID.init_from_data(
            y, 
            shape=-0.2, 
            threshold=np.quantile(ds_station.t2m_max, q=0.95)
            # scale=2.0,
        )

        pbar.set_description(f"Station ID: {istation_id} | Initializing MAP Model...")

        method = "laplace"
        num_steps = 10_000
        step_size = 1e-3
        clip_norm = 0.1
        num_samples = 10

        learner = SVILearner(
            model=model.model,
            method=method,
            num_steps=num_steps,
            step_size=step_size,
            clip_norm=clip_norm,
            num_samples=num_samples,
            
            )

        pbar.set_description(f"Station ID: {istation_id} | Calculating MAP Results...")
        rng_key_train, rng_key = jrandom.split(rng_key, num=2)
        svi_post = learner(rng_key, y=y)
        params = svi_post.median_params


        pbar.set_description(f"Station ID: {istation_id} | Calculating MCMC Results...")
        num_samples = 5_000
        num_warmup = 2_000
        mcmc_learner = MCMCLearner(
            model=model.model,
            num_samples=num_samples,
            num_warmup = num_warmup,
            init_params = params
            )

        # create key
        rng_key_train, rng_key = jrandom.split(rng_key, num=2)

        mcmc_post = mcmc_learner(rng_key_train, y=y)

        # correct coordinates
        pbar.set_description(f"Station ID: {istation_id} | Creating Posterior DS...")
        arxiv_summary = calculate_posterior(mcmc_post.mcmc, model, rng_key)
        arxiv_summary = arxiv_summary.assign_coords({"time": ds_bm.time})
        arxiv_summary = arxiv_summary.assign_coords({"return_period": model.return_periods})

        pbar.set_description(f"Station ID: {istation_id} | Adding MAP params...")
        arxiv_summary.posterior["threshold"] = threshold.squeeze()
        arxiv_summary.posterior["map_shape"] = params["concentration"].squeeze()
        arxiv_summary.posterior["map_scale"] = params["scale"].squeeze()
        arxiv_summary.posterior["map_location"] = params["location"].squeeze()

        # adding statistics
        pbar.set_description(f"Station ID: {istation_id} | Adding Metrics...")
        out = az.waic(arxiv_summary, )
        arxiv_summary.log_likelihood["elpd_waic"] = out.elpd_waic
        arxiv_summary.log_likelihood["elpd_waic_se"] = out.se
        arxiv_summary.log_likelihood["p_waic"] = out.p_waic

        pbar.set_description(f"Station ID: {istation_id} | Saving DS...")

        results_save_paths = MySavePaths(
            base_path=MY_ROOT_PATHS.data_results_dir,
            stage=stage,
            method=method_name,
            region=str(ds_station.station_id.values.squeeze())
            )
        
        results_save_paths.make_dir()

        # save_name = Path(figures_save_dir).joinpath(f"{savefilename}").joinpath("results")
        # save_name.mkdir(parents=True, exist_ok=True)
        arxiv_summary.to_netcdf(results_save_paths.full_path.joinpath("results.zarr"), engine="netcdf4")
        
        # PLOTS
        pbar.set_description(f"Station ID: {istation_id} | Plotting Parameter Traces...")
        var_names = ["location", "scale", "concentration", "rate", "return_level_100"]
        plot_posterior_param_trace(arxiv_summary, var_names, fig_save_path)
        plot_posterior_predictive(arxiv_summary, fig_save_path)
        plot_posterior_params_joint(arxiv_summary, var_names, fig_save_path)
        plot_qq_plot_gevd(arxiv_summary, fig_save_path)

        # calculate return period
        pbar.set_description(f"Station ID: {istation_id} | Plot Return Level...")
        plot_return_level_gevd(arxiv_summary=arxiv_summary, model=model, y=y, figures_save_dir=fig_save_path)
        plot_return_level_hist(arxiv_summary=arxiv_summary, figures_save_dir=fig_save_path)
            

if __name__ == '__main__':
    app()