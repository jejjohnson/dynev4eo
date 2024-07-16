import autoroot
import typer
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from loguru import logger
from dynev4eo._src.io import MyPaths
from dynev4eo._src.preprocess.masks import add_country_mask
from dynev4eo._src.preprocess.validation import validate_longitude, validate_latitude
import pint_xarray



app = typer.Typer()


@app.command()
def clean_max_temp_stations(feten: bool=False):
    logger.info(f"Starting script...")

    logger.info(f"Sorting out paths...")
    my_paths = MyPaths.init_from_dot_env()

    logger.info(f"Loading station coordinates...")
    df_coords = pd.read_csv(my_paths.data_raw_dir.joinpath("ubicacion_estaciones_spain.csv"), delimiter=";", index_col=0, decimal=",")

    logger.info(f"Loading max temperature values...")
    df_all = pd.read_csv(my_paths.data_raw_dir.joinpath("tmax_homo.csv"), index_col=0)

    if feten:
        logger.info(f"Loading **good** stations...")
        red_feten_stations = pd.read_csv(my_paths.data_raw_dir.joinpath("red_feten.csv"))

        # get intersection of stations
        tmax_red_feten_stations = np.intersect1d(red_feten_stations.id, df_all.columns)
        df_all = df_all.loc[:, tmax_red_feten_stations]

    coordinates = dict(
        station_id=list(),
        station_name=list(),
        lat=list(),
        lon=list(),
        alt=list(),
        values=list()
    )

    logger.info(f"Creating xarray datastructure...")
    xr_datasets = xr.Dataset()
    pbar = tqdm(df_all.columns, leave=False)
    for iname in pbar:

        

        try:
            ids = df_all[str(iname)]
            icoords = df_coords.loc[str(iname)]
            # extract coordinates
            coordinates["station_id"].append(icoords.name)
            coordinates["station_name"].append(icoords["name"].lower())
            coordinates["lat"].append(np.float32(icoords["lat"]))
            coordinates["lon"].append(np.float32(icoords["lon"]))
            coordinates["alt"].append(np.float32(icoords["alt"]))
            coordinates["values"].append(ids.values)
        except KeyError:
            pass

    ds_tmax = xr.Dataset(
        {
            "t2m_max": (("station_id", "time"), coordinates['values']),
            "lon": (("station_id"), coordinates['lon']),
            "lat": (("station_id"), coordinates['lat']),
            "alt": (("station_id"), coordinates['alt']),
            "station_name": (("station_id"), coordinates['station_name']),
        },
        coords={
            "station_id": coordinates["station_id"],
            "time": pd.to_datetime(df_all.index.values)
        }
    )

    logger.info(f"Cleaning metadata and coordinates...")

    # assign coordinates
    ds_tmax = ds_tmax.set_coords(["lon", "lat", "alt", "station_name"])

    # valudate coordinates
    ds_tmax = validate_longitude(ds_tmax)
    ds_tmax = validate_latitude(ds_tmax)

    ds_tmax = ds_tmax.sortby("time")


    ds_tmax["t2m_max"].attrs["standard_name"] = "2m_temperature_max"
    ds_tmax["t2m_max"].attrs["long_name"] = "2m Temperature Max"


    ds_tmax["alt"].attrs["standard_name"] = "altitude"
    ds_tmax["alt"].attrs["long_name"] = "Altitude"

    # # validate units
    # ds_tmax["lon"].attrs["units"] = "degree"
    # ds_tmax["lat"].attrs["units"] = "degree"
    # ds_tmax = ds_tmax.pint.dequantify()
    ds_tmax = ds_tmax.pint.quantify(
        {"t2m_max": "degC", 
        "lon": "degree", 
        "lat": "degree",
        "alt": "meters"
        }
    )
    ds_tmax = ds_tmax.pint.dequantify()
    # rename variable

    logger.info(f"Adding country mask...")
    ds_tmax = add_country_mask(ds_tmax, "Spain") 

    #
    logger.info(f"Saving data to disk...")

    if feten:
        save_name = "t2m_stations_feten_spain.zarr"
    else:
        save_name = "t2m_stations_spain.zarr"
    
    full_save_path = my_paths.data_clean_dir.joinpath(save_name)

    logger.info(f"Checking directory...")
    logger.debug(f"Saving to {full_save_path}")
    assert full_save_path.parent.is_dir()

    ds_tmax.to_zarr(full_save_path, mode="w")

    logger.info(f"Finished script...")



if __name__ == '__main__':
    app()
