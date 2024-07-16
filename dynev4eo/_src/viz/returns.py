from pathlib import Path
from typing import List, Optional
import numpy as np
import arviz as az
import jax
import matplotlib.pyplot as plt
import seaborn as sns
sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)
from dynev4eo._src.extremes.returns import calculate_exceedence_probs
from matplotlib.ticker import ScalarFormatter


def plot_return_level_gevd(arxiv_summary, model, y, figures_save_dir):

    return_period_empirical = 1 / calculate_exceedence_probs(y) 

    fig, ax = plt.subplots(figsize=(6.5, 6))

    ciu, mu, cil = arxiv_summary.posterior_predictive["return_level"].quantile(q=[0.025, 0.5, 0.975], dim=["chain", "draw"])

    ax.plot(
        model.return_periods, mu, 
        linestyle="-", linewidth=3, color="tab:red",
        label="GEVD (Mean)"
    )

    ax.fill_between(
        model.return_periods, 
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
        x=return_period_empirical,
        y=y,
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
    if figures_save_dir is not None:
        plt.savefig(Path(figures_save_dir).joinpath(f"returns_prob_posterior_vs_empirical.png")) 
        plt.close()
    else:
        return fig, ax



def plot_return_level_gpd(arxiv_summary, figures_save_dir):


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
    if figures_save_dir is not None:
        plt.savefig(Path(figures_save_dir).joinpath(f"returns_prob_posterior_vs_empirical.png")) 
        plt.close()
    else:
        return fig, ax



def plot_return_level_hist(arxiv_summary, figures_save_dir):
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
    if figures_save_dir is not None:
        fig.savefig(Path(figures_save_dir).joinpath("returns_100yr_hist.png")) 
        plt.close()
    else:
        return fig, ax

