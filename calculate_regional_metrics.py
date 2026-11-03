"""Calculate regional SIT metrics for the 2022--2023 freezing-season files.

Regions are non-overlapping geographic polygons defined on the native PIOMAS
120 x 360 latitude/longitude grid.  Land and invalid cells are excluded using
the finite-value mask in each daily NetCDF file.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parent
NC_DIR = ROOT / "nc_files"
LAT_FILE = Path("/workspace/scratch/0b6741d41344/upload/latgrid.dat")
LON_FILE = Path("/workspace/scratch/0b6741d41344/upload/longrid.dat")

# Geographic masks (degrees north/east). Boundaries are intentionally
# non-overlapping, so each grid cell is assigned to no more than one region.
REGIONS = {
    "Beaufort Sea": {"lat_min": 68.0, "lat_max": 82.0, "lon_min": -160.0, "lon_max": -125.0},
    "Chukchi Sea": {"lat_min": 66.0, "lat_max": 78.0, "lon_min": -180.0, "lon_max": -160.0},
    "East Siberian Sea": {"lat_min": 68.0, "lat_max": 80.0, "lon_min": 140.0, "lon_max": 180.0},
    "Laptev Sea": {"lat_min": 70.0, "lat_max": 82.0, "lon_min": 105.0, "lon_max": 140.0},
    "Kara Sea": {"lat_min": 68.0, "lat_max": 82.0, "lon_min": 50.0, "lon_max": 105.0},
    "Barents Sea": {"lat_min": 68.0, "lat_max": 82.0, "lon_min": 15.0, "lon_max": 50.0},
    "Greenland Sea": {"lat_min": 68.0, "lat_max": 82.0, "lon_min": -45.0, "lon_max": 15.0},
    "Baffin Bay": {"lat_min": 65.0, "lat_max": 78.0, "lon_min": -80.0, "lon_max": -50.0},
    "Canadian Arctic Archipelago": {"lat_min": 68.0, "lat_max": 82.0, "lon_min": -125.0, "lon_max": -80.0},
    "Central Arctic Ocean": {"lat_min": 82.0, "lat_max": 90.0, "lon_min": -180.0, "lon_max": 180.0},
}


def main() -> None:
    lat = np.loadtxt(LAT_FILE).reshape(120, 360)
    lon = np.loadtxt(LON_FILE).reshape(120, 360)
    # The supplied PIOMAS longitude grid uses 0--360 degrees east. Convert it
    # to the -180--180 convention used by the regional boundaries below.
    lon = ((lon + 180.0) % 360.0) - 180.0
    files = sorted(NC_DIR.rglob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {NC_DIR}")

    masks = {}
    for name, b in REGIONS.items():
        masks[name] = (
            (lat >= b["lat_min"]) & (lat < b["lat_max"]) &
            (lon >= b["lon_min"]) & (lon < b["lon_max"])
        )

    # Sum-of-squares formulation yields exact pooled metrics without retaining
    # millions of values in memory. R2 uses the reference-field mean over all
    # valid regional daily samples.
    acc = {name: dict(n=0, sum_obs=0.0, sum_obs2=0.0,
                      sum_abs_bc=0.0, sum_sq_bc=0.0,
                      sum_abs_piomas=0.0, sum_sq_piomas=0.0)
           for name in REGIONS}

    for f in files:
        with Dataset(f) as ds:
            piomas = np.asarray(ds.variables["nobc_thickness"][:].filled(np.nan), dtype=float)
            bc = np.asarray(ds.variables["bc_thickness"][:].filled(np.nan), dtype=float)
            obs = np.asarray(ds.variables["obs"][:].filled(np.nan), dtype=float)
        valid0 = np.isfinite(piomas) & np.isfinite(bc) & np.isfinite(obs)
        for name, region_mask in masks.items():
            valid = valid0 & region_mask
            if not valid.any():
                continue
            p, c, o = piomas[valid], bc[valid], obs[valid]
            e_bc, e_piomas = c - o, p - o
            a = acc[name]
            a["n"] += o.size
            a["sum_obs"] += o.sum()
            a["sum_obs2"] += np.square(o).sum()
            a["sum_abs_bc"] += np.abs(e_bc).sum()
            a["sum_sq_bc"] += np.square(e_bc).sum()
            a["sum_abs_piomas"] += np.abs(e_piomas).sum()
            a["sum_sq_piomas"] += np.square(e_piomas).sum()

    result_rows = []
    for name, b in REGIONS.items():
        a = acc[name]
        n = a["n"]
        if n == 0:
            raise RuntimeError(f"No valid samples in {name}")
        sst = a["sum_obs2"] - a["sum_obs"] ** 2 / n
        result_rows.append({
            "Region": name,
            "Lat_range_degN": f'[{b["lat_min"]}, {b["lat_max"]})',
            "Lon_range_degE": f'[{b["lon_min"]}, {b["lon_max"]})',
            "N_valid_grid-day_samples": n,
            "PIOMAS_RMSE_m": np.sqrt(a["sum_sq_piomas"] / n),
            "PIOMAS_MAE_m": a["sum_abs_piomas"] / n,
            "PIOMAS_R2": 1 - a["sum_sq_piomas"] / sst,
            "SIT_BCNet_RMSE_m": np.sqrt(a["sum_sq_bc"] / n),
            "SIT_BCNet_MAE_m": a["sum_abs_bc"] / n,
            "SIT_BCNet_R2": 1 - a["sum_sq_bc"] / sst,
        })

    fieldnames = list(result_rows[0])
    out = ROOT / "regional_performance_2022_2023_freezing_season.csv"
    with out.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    # Save reproducible masks and coordinates.
    np.savez_compressed(ROOT / "arctic_basin_masks_120x360.npz", latitude=lat,
                        longitude=lon, **{name.replace(" ", "_"): mask.astype(np.uint8)
                                             for name, mask in masks.items()})
    with (ROOT / "regional_mask_definitions.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["Region", "Lat_min_degN", "Lat_max_degN", "Lon_min_degE", "Lon_max_degE"])
        writer.writeheader()
        for name, b in REGIONS.items():
            writer.writerow({"Region": name, "Lat_min_degN": b["lat_min"], "Lat_max_degN": b["lat_max"],
                             "Lon_min_degE": b["lon_min"], "Lon_max_degE": b["lon_max"]})
    print(out)


if __name__ == "__main__":
    main()
