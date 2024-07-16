from pathlib import Path
from typing import List, Optional
import numpy as np
import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)
from tensorflow_probability.substrates.jax import distributions as tfd




def plot_qq_plot_gevd(arxiv_summary, figures_save_dir):
    y_min = arxiv_summary.observed_data.obs.min().values
    y_max = arxiv_summary.observed_data.obs.max().values

    diff = 0.10 * (y_max - y_min)
    y_domain = np.linspace(y_min - diff, y_max + diff, 100)



    fn_quantile = lambda x: tfd.GeneralizedExtremeValue(
        loc=arxiv_summary.posterior["location"].squeeze(),
        scale=arxiv_summary.posterior["scale"].squeeze(),
        concentration=arxiv_summary.posterior["concentration"].squeeze(),
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
    if figures_save_dir is not None:
        fig.savefig(Path(figures_save_dir).joinpath(f"qqplot.png")) 
        plt.close()
    else:
        return fig, ax

def plot_qq_plot_gpd(arxiv_summary, figures_save_dir=None):
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
    if figures_save_dir is not None:
        fig.savefig(Path(figures_save_dir).joinpath(f"qqplot.png")) 
        plt.close()
    else:
        return fig, ax