from pathlib import Path
from typing import List, Optional
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


def plot_posterior_predictive(arxiv_summary, figures_save_dir):
    fig, ax = plt.subplots()
    fig = az.plot_ppc(arxiv_summary, ax=ax, group="posterior", colors=["tab:blue", "tab:red", "black"])
    ax.set(
        xlabel="Temperature [°C]",
        # ylabel="Density",
    )
    plt.tight_layout()
    if figures_save_dir is not None:
        plt.savefig(Path(figures_save_dir).joinpath(f"pred_prob_posterior.png")) 
        plt.close()
    else:
        return fig