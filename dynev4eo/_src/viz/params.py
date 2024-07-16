from pathlib import Path
from typing import List, Optional
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


def plot_posterior_param_trace(arxiv_summary, var_names: List[str], figures_save_dir):

    fig = az.plot_trace(
        arxiv_summary, 
        var_names=var_names,);
    plt.tight_layout()
    if figures_save_dir is not None:
        plt.savefig(Path(figures_save_dir).joinpath(f"param_divergences.png")) 
        plt.close()
    else:
        return fig
    

def plot_posterior_params_joint(arxiv_summary, var_names: List[str], figures_save_dir):
    fig = az.plot_pair(
        arxiv_summary,
        group="posterior",
        var_names=var_names,
        kind=["scatter", "kde"],
        kde_kwargs={"fill_last": False},
        marginals=True,
        # coords=coords,
        point_estimate="median",
        figsize=(11.5, 10),
    )
    if figures_save_dir is not None:
        plt.savefig(Path(figures_save_dir).joinpath(f"params_prob_posterior.png")) 
        plt.close()
    else:
        return fig
