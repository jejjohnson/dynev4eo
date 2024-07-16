from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


def plot_ts_station_bm(ds_station, ds_bm, figures_save_dir: Optional[str]=None):

    fig, ax = plt.subplots(figsize=(10,5))
    ds_bm.plot.scatter(x="time", y="t2m_max", ax=ax, color="tab:red", s=30.0, zorder=3)
    ds_station.plot.scatter(x="time", y="t2m_max", ax=ax, color="tab:blue", s=2.0, zorder=1, facecolors='none', edgecolors='tab:blue', linewidths=1.0)
    ax.set(
        xlabel="Time [Days]",
        title=str(ds_bm.station_name.values).upper(),
    )
    ax.xaxis_date()
    plt.tight_layout()
    if figures_save_dir:
        fig.savefig(Path(figures_save_dir).joinpath(f"{ds_bm.station_id.values}_t2mmax_bm_ts.png"))
        plt.close()
    else:
        return fig, ax
    

def plot_ts_station_pot(ds_station, ds_pot, figures_save_dir: Optional[str]=None):

    fig, ax = plt.subplots(figsize=(10,5))
    ds_pot.plot.scatter(x="time", y="t2m_max", ax=ax, color="tab:red", s=30.0, zorder=3)
    ax.hlines(ds_pot.threshold, xmin=ds_station.time.min(), xmax=ds_station.time.max(), color="black", linewidth=3, zorder=2, label=f"Threshold - {ds_pot.threshold.item():.0f}")
    ds_station.plot.scatter(x="time", y="t2m_max", ax=ax, color="tab:blue", s=2.0, zorder=1, facecolors='none', edgecolors='tab:blue', linewidths=1.0)
    ax.set(
        xlabel="Time [Days]",
        title=str(ds_pot.station_name.values).upper(),
    )
    ax.xaxis_date()
    plt.tight_layout()
    if figures_save_dir:
        fig.savefig(Path(figures_save_dir).joinpath(f"{ds_pot.station_id.values}_t2mmax_pot_ts.png"))
        plt.close()
    else:
        return fig, ax

    

def plot_scatter_station(ds_bm, figures_save_dir: Optional[str]=None):
    fig, ax = plt.subplots(figsize=(10,5))

    ds_bm.t2m_max.plot.scatter(ax=ax, x="time", zorder=3, color="tab:red", facecolors="none", linewidths=3)
    ax.set(
        xlabel="Time [Years]",
        title=f"{str(ds_bm.station_name.values).upper()}",
    )
    plt.tight_layout()
    if figures_save_dir:
        fig.savefig(Path(figures_save_dir).joinpath(f"{ds_bm.station_id.values}_t2mmax_bm_scatter.png"))
        plt.close()
    else:
        return fig, ax
    

def plot_histogram_station(ds_bm, figures_save_dir: Optional[str]=None):
    fig, ax = plt.subplots()
    ds_bm.t2m_max.plot.hist(ax=ax, bins=20, linewidth=4, density=False, fill=False)
    ax.set(
        ylabel="Number of Observations",
        title=f"{str(ds_bm.station_name.values).upper()}",
    )
    plt.tight_layout()
    if figures_save_dir:
        fig.savefig(Path(figures_save_dir).joinpath(f"{ds_bm.station_id.values}_t2mmax_bm_hist.png"))
        plt.close()
    else:
        return fig, ax