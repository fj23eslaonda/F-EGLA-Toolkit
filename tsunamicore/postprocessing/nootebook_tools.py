#--------------------------------------------------------
#
# PACKAGES
#
#--------------------------------------------------------
# --- Standard libraries ---
import pickle
from pathlib import Path

# --- Numerical and data handling ---
import numpy as np
import pandas as pd

# --- Plotting ---
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon

# --- xarray for bathymetry ---
import xarray as xr

# --- Progress bar ---
from tqdm import tqdm

# --- GIS & geometry ---
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform
import geopandas as gpd
from pyproj import Transformer
transformer_utm = Transformer.from_crs("epsg:4326", "epsg:32719", always_xy=True)
transformer_wgs = Transformer.from_crs("epsg:32719", "epsg:4326", always_xy=True)

# --- KMZ export ---
import simplekml
from simplekml import Color

# --- FEGLA core imports ---
from tsunamicore.fegla.model import process_transect
from tsunamicore.fegla.operations import transect_processing

# --- Style ---
from tsunamicore.utils.plot_style import apply_plot_style
apply_plot_style()


#--------------------------------------------------------
#
# FUNCTIONS
#
#--------------------------------------------------------
def run_FEGLA(transectData, params, production_path, tolerance=0.01, verbose=True):
    """
    Runs the FEGLA model for multiple user-defined hmax values.
    For each hmax, all transects are processed using the calibrated model
    (bestF0, bestFR, bestmodel), and convergence diagnostics are computed.

    Output:
        results[hmax]     -> FEGLA results dict for each transect
        error_dict[hmax]  -> error dict for each transect
    """

    # -----------------------------
    # Prepare inputs & validation
    # -----------------------------
    production_path = Path(production_path)
    production_path.mkdir(parents=True, exist_ok=True)

    city      = params['city']
    manning   = params['manning']
    mode      = params['bestmodel']
    F0        = params['bestF0']
    FR        = params['bestFR'] if params['bestFR'] is not None else params['bestF0']
    hmax_list = params['hmax']       # NOW A LIST

    transect_keys = list(transectData.keys())

    if verbose:
        print(f"\nRunning FEGLA for city: {city}")
        print(f"Mode: {mode},  F0={F0},  FR={FR}")
        print(f"Testing {len(hmax_list)} hmax scenarios: {hmax_list}")
        print(f"Number of transects: {len(transect_keys)}\n")

    # -----------------------------
    # Output containers
    # -----------------------------
    results_all = {}
    errors_all  = {}

    # =====================================================
    # LOOP OVER EACH HMAX
    # =====================================================
    for hmean in hmax_list:

        if verbose:
            print(f"\n==========================================")
            print(f" Running FEGLA for hmax = {hmean:.3f} m")
            print(f"==========================================\n")

        results = {}
        error_dict = {}

        # -----------------------------
        # Run FEGLA per transect
        # -----------------------------
        for idx in tqdm(transect_keys, desc=f"hmax = {hmean:.2f} m"):
            df = transectData[idx]

            # Preprocess transect
            df_processed = transect_processing(
                {idx: df}, 
                [idx],
                manning_coeff=manning,
                flood=False
            )[idx]

            # Run FEGLA model
            out_idx, out_res = process_transect(
                idx,
                {idx: df_processed},
                F0=F0,
                FR=FR,
                mode=mode,
                hmean_value=hmean,
                message=False
            )

            results[out_idx] = out_res
            error_dict[out_idx] = out_res['error']

        # -----------------------------
        # Compute convergence statistics
        # -----------------------------
        errors = np.array(list(error_dict.values()))
        converged_mask = errors < tolerance

        n_converged = np.sum(converged_mask)
        n_total     = len(errors)
        percent     = 100 * n_converged / n_total

        # Non-converged errors
        nonconv_errors = errors[~converged_mask]
        n_nonconv      = len(nonconv_errors)

        if verbose:
            print(f"{n_converged} out of {n_total} transects "
                  f"({percent:.1f}%) converged below {tolerance} m.")

            if n_nonconv > 0:
                err_min = float(np.nanmin(nonconv_errors))
                err_max = float(np.nanmax(nonconv_errors))
                print(f"{n_nonconv} transects did not converge.")
                print(f"Non-converged errors range: {err_min:.3f} → {err_max:.3f} m\n")
            else:
                print("All transects converged.\n")

        # -----------------------------
        # Store results for this hmax
        # -----------------------------
        results_all[hmean] = results
        errors_all[hmean]  = error_dict

        # -----------------------------
        # Save results as .pkl
        # -----------------------------
        fname = (
            f"{city}_hmax_{hmean:.2f}m_{mode}_F0_{F0}"
            + (f"_FR_{FR}" if params['bestFR'] is not None else "")
            + ".pkl"
        )

        fpath = production_path / fname

        with open(fpath, "wb") as f:
            pickle.dump(results, f)

        if verbose:
            print(f"Saved FEGLA results for hmax={hmean:.2f} m:")
            print(f" → {fpath}\n")

    # -----------------------------
    # RETURN
    # -----------------------------
    return results_all, errors_all

def merge_height(transectData, results, col_name="height"):
    """
    Merge FEGLA 'height' results into each transect DataFrame.

    Parameters
    ----------
    transectData : dict
        Dictionary of DataFrames for each transect (e.g., 'T001', 'T002', ...)
    results : dict
        Dictionary of FEGLA outputs where results[transect_id]['height'] exists.
    col_name : str
        Name of the column to add to each DataFrame.

    Returns
    -------
    dict
        Updated transectData with a new column containing aligned FEGLA heights.
    """

    merged = {}

    for key, df in transectData.items():

        df = df.copy()

        if key not in results:
            print(f"Warning: no FEGLA result found for {key}. Filling with NaN.")
            df[col_name] = np.nan
            merged[key] = df
            continue

        # FEGLA height list
        height = np.array(results[key]["height"], dtype=float)

        n_df = len(df)
        n_h = len(height)

        # Align lengths
        if n_h > n_df:
            height_aligned = height[:n_df]  # truncate
        elif n_h < n_df:
            pad = np.full(n_df - n_h, np.nan)
            height_aligned = np.concatenate([height, pad])
        else:
            height_aligned = height

        # Add new column
        df[col_name] = height_aligned
        merged[key] = df

    return merged


def calculate_polygon_and_area(first_points, last_points):
    """
    Optimized version:
    - Avoids GeoDataFrame and repeated to_crs transformations.
    - Uses pyproj Transformer directly for fast coordinate projection.
    """
    # Combine points into one array
    points = np.vstack([first_points, last_points[::-1]])
    
    # Transform coordinates to UTM
    x, y = transformer_utm.transform(points[:, 0], points[:, 1])
    
    # Create polygon in projected coordinates
    polygon_proj = Polygon(np.column_stack((x, y)))
    
    # Calculate area in square meters directly
    area_m2 = polygon_proj.area
    
    return polygon_proj, area_m2


def get_boundary_points(data, column):
    """
    Extracts boundary (first / last) points from each flooded transect and computes
    a polygon representing the inundation envelope.

    Parameters
    ----------
    data : dict[str, pandas.DataFrame]
        Dictionary where keys are transect IDs and each value is a DataFrame with
        'lon', 'lat', and the water column variable.
    column : str
        Column name containing inundation height or depth (e.g., 'height').

    Returns
    -------
    polygonSim : shapely.geometry.Polygon or None
        Polygon describing the inundated area.
    aream2Sim : float or None
        Surface area of the polygon (m²), projected to UTM.
    """

    first_points = []
    last_points  = []

    for key, df in data.items():
        # Remove rows where the water column is NaN
        df_valid = df.dropna(subset=[column])

        # Must have at least two valid points
        if df_valid.empty or len(df_valid) < 2:
            continue

        # Extract first and last valid lon/lat
        first_points.append((df_valid['lon'].iloc[0], df_valid['lat'].iloc[0]))
        last_points.append((df_valid['lon'].iloc[-1], df_valid['lat'].iloc[-1]))

    # Safety check
    if len(first_points) < 2 or len(last_points) < 2:
        print(f"[WARNING] Not enough transects with valid {column} values.")
        return None, None

    first_arr = np.array(first_points)
    last_arr  = np.array(last_points)

    # Build polygon from these boundaries
    polygonSim, aream2Sim = calculate_polygon_and_area(first_arr, last_arr)

    return polygonSim, aream2Sim


def plot_flood_maps(
    city: str,
    data_path: Path,
    transectData: dict,
    results_all: dict,              # {hmax_value : results_dict}
    transformer_wgs,
    production_path: Path,
    levels=None,
    figsize=(9, 9),
):
    """
    Plot bathymetry + FEGLA flooded polygon for each hmax value.
    Saves PNGs and returns polygons + areas for each scenario.

    Parameters
    ----------
    city : str
        Study site name.
    data_path : Path
        Path to /data/<city> containing Bathymetry.nc.
    transectData : dict
        Dictionary of transect DataFrames.
    results_all : dict
        Dictionary mapping hmax -> FEGLA results.
    transformer_wgs : Transformer
        Transformer from UTM → WGS84.
    production_path : Path
        Output folder: (...)/outputs/<city>/production
    levels : list, optional
        Contour levels for bathymetry.
    figsize : tuple
        Figure size.

    Returns
    -------
    polygons_dict : dict
        {hmax_value : shapely_polygon_in_utm}
    areas_dict : dict
        {hmax_value : area_m2}
    """

    figs_path = production_path / "figs"
    figs_path.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------
    # Load bathymetry
    # ----------------------------------------------------
    bathydata = xr.open_dataset(data_path / "Bathymetry.nc")
    lat = bathydata["lat"].values
    lon = bathydata["lon"].values
    bathy = bathydata["bathy"].values

    # ----------------------------------------------------
    # Default contour levels
    # ----------------------------------------------------
    if levels is None:
        levels = [-100, -50, -30, -20, -10, 0, 5, 10, 20, 30, 50, 100]

    polygons_dict = {}
    areas_dict = {}

    # ====================================================
    # LOOP THROUGH ALL HMAX VALUES
    # ====================================================
    for hmax_value, results in results_all.items():

        # ---------------------------------------------
        # Merge FEGLA heights into transectData
        # ---------------------------------------------
        tData = merge_height(transectData, results)

        # ---------------------------------------------
        # Compute polygon (in UTM)
        # ---------------------------------------------
        polygon_utm, area_m2 = get_boundary_points(tData, "height")
        polygons_dict[hmax_value] = polygon_utm
        areas_dict[hmax_value] = area_m2

        # ---------------------------------------------
        # Convert polygon UTM → WGS84
        # ---------------------------------------------
        polygon_wgs = transform(transformer_wgs.transform, polygon_utm)
        poly_x, poly_y = polygon_wgs.exterior.xy

        # ---------------------------------------------
        # Build custom discrete colormap (bathy + topo)
        # ---------------------------------------------
        i0 = levels.index(0)
        k_neg = i0
        k_pos = len(levels) - 1 - i0

        bathy_cmap = plt.cm.GnBu
        topo_cmap  = plt.cm.gist_gray_r

        blues = bathy_cmap(np.linspace(0.2, 1, k_neg)) if k_neg > 0 else np.empty((0,4))
        grays = topo_cmap(np.linspace(0.2, 1, k_pos)) if k_pos > 0 else np.empty((0,4))
        colors = np.vstack([blues, grays])

        listed = mcolors.ListedColormap(colors)
        listed.set_under(colors[0])
        listed.set_over(colors[-1])

        norm = mcolors.BoundaryNorm(levels, ncolors=listed.N, clip=True)

        # ---------------------------------------------
        # Plot figure
        # ---------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)

        cf = ax.contourf(
            lon, lat, bathy,
            levels=levels,
            cmap=listed,
            norm=norm,
            alpha=0.55,
            extend="both",
        )

        cs = ax.contour(
            lon, lat, bathy,
            levels=levels,
            colors="black",
            linewidths=0.5,
        )
        ax.clabel(cs, inline=True, fontsize=10, fmt="%.0f")

        cbar = fig.colorbar(
            cf, ax=ax,
            extend="both",
            fraction=0.08,
            pad=0.04
        )
        cbar.set_label("Topobathymetry [m]")
        cbar.locator = plt.FixedLocator(levels)
        cbar.update_ticks()

        # Flood polygon (fill + outline)
        poly_patch = MplPolygon(
            np.column_stack((poly_x, poly_y)),
            closed=True,
            facecolor="#ff7300",
            edgecolor="#ff7300",
            linewidth=2,
            alpha=0.25
        )
        ax.add_patch(poly_patch)
        ax.plot(poly_x, poly_y, c="#ff7300", ls="-", linewidth=2, label="Flooded area")

        # ---------------------------------------------
        # Title and formatting
        # ---------------------------------------------
        area_km2 = area_m2 / 1e6
        ax.set_title(
            f"Estimated flooded area: {area_km2:.2f} km²\nfor Hmax = {hmax_value:.2f} m",
            fontsize=14
        )

        ax.set_aspect("equal")
        ax.set_xlabel("Longitude [°]")
        ax.set_ylabel("Latitude [°]")
        ax.grid(ls="--", lw=1, alpha=0.15)
        ax.legend(loc="best")

        plt.tight_layout()
        plt.show()

        # ---------------------------------------------
        # Save figure
        # ---------------------------------------------
        figname = f"{city}_Hmax_{hmax_value:.2f}m.png"
        fig.savefig(figs_path / figname, dpi=200)
        plt.close(fig)

    return polygons_dict, areas_dict

def save_all_flood_polygons(
    city: str,
    polygons: dict,               # {hmax_value : polygon_utm}
    production_path: Path,
    fmt=("kmz", "shp"),           # save KMZ and SHP by default
    source_crs: str | int = "EPSG:32719",   # UTM zone (your polygons are UTM!)
    target_crs: str | int = "EPSG:4326",
    color: str = "#ff7300"       # red for flooded polygon
):
    """
    Save all flooded-area polygons (one per hmax) as KMZ and/or SHP
    into:
        production/kmz/
        production/shapefile/

    Naming follows figure convention:
        City_Hmax_Xm.kmz
        City_Hmax_Xm.shp
    """
    # -----------------------------------------------
    # Create folders
    # -----------------------------------------------
    kmz_dir = production_path / "kmz"
    shp_dir = production_path / "shapefile"

    kmz_dir.mkdir(parents=True, exist_ok=True)
    shp_dir.mkdir(parents=True, exist_ok=True)

    # Convert fmt arg to list
    if isinstance(fmt, str):
        fmt = [fmt.lower()]
    else:
        fmt = [f.lower() for f in fmt]

    # -----------------------------------------------
    # Loop over all polygons (each hmax)
    # -----------------------------------------------
    print("\nSaving flood polygons...\n")

    for hmax_value, poly in tqdm(polygons.items(), desc='Saving flooded areas: '):

        # Name used in PNG:
        #    City_Hmax_Xm.png
        base_name = f"{city}_Hmax_{hmax_value:.2f}m"

        # Convert polygon → MultiPolygon object
        geom = _to_multipolygon(_as_geom(poly))

        # -------------------------------------------
        # Save SHP
        # -------------------------------------------
        if "shp" in fmt:
            gdf = gpd.GeoDataFrame(
                [{"hmax": hmax_value, "geometry": geom}],
                geometry="geometry",
                crs=source_crs
            )
            gdf = gdf.to_crs(target_crs)

            shp_path = shp_dir / f"{base_name}.shp"
            gdf.to_file(shp_path)

            #print(f"Saved SHP → {shp_path}")

        # -------------------------------------------
        # Save KMZ
        # -------------------------------------------
        if "kmz" in fmt:
            kml = simplekml.Kml()
            folder = kml.newfolder(name=base_name)

            # Convert CRS
            gdf = gpd.GeoDataFrame(
                [{"geometry": geom}],
                geometry="geometry",
                crs=source_crs
            ).to_crs(target_crs)

            mp = gdf.geometry.iloc[0]
            polys = mp.geoms if isinstance(mp, MultiPolygon) else [mp]

            # Convert hex → KML color
            r, g, b, a = _hex_to_rgba(color, a=160)  
            kml_color = Color.rgb(r, g, b, a=a)

            for p in polys:
                pol = folder.newpolygon(name=base_name)
                x, y = p.exterior.coords.xy
                pol.outerboundaryis = list(zip(x, y))

                # Holes
                for ring in p.interiors:
                    xr, yr = ring.coords.xy
                    pol.innerboundaryis.append(list(zip(xr, yr)))

                pol.style.polystyle.color = kml_color
                pol.style.linestyle.color = kml_color
                pol.style.linestyle.width = 2.0

            kmz_path = kmz_dir / f"{base_name}.kmz"
            kml.savekmz(str(kmz_path))

            #print(f"Saved KMZ → {kmz_path}")

    print("\nAll flood polygons exported successfully!\n")

def _as_geom(obj):
    """
    Safely convert an object into a shapely geometry.
    Accepts a shapely Polygon, MultiPolygon, or an iterable of coordinate pairs.
    """
    if isinstance(obj, (Polygon, MultiPolygon)):
        return obj
    try:
        return Polygon(obj)
    except Exception:
        raise ValueError(f"Cannot convert object to geometry: {obj}")


def _to_multipolygon(geom):
    """
    Ensures the geometry is always a shapely MultiPolygon.
    """
    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    raise ValueError(f"Cannot convert to MultiPolygon: {geom}")

def _hex_to_rgba(hex_color, a=255):
    """
    Convert a hex color string (#RRGGBB) into (r, g, b, a) integers.
    Alpha defaults to 255 unless specified (e.g., a=160 for semi-transparent).
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b, a

def load_transects(transect_path: Path):
    """
    Load all transects from a single HDF5 file and return a dict.
    Assumes file name: 'Transects_data.h5'

    Returns:
        dict: { 'T001': df, 'T002': df, ... }
    """
    h5_file = transect_path / "Transects_data.h5"

    print("Loading transects...")

    with pd.HDFStore(h5_file, mode="r") as store:
        transects = {
            key.lstrip("/") : store[key]
            for key in store.keys()
        }

    print(f"Loaded {len(transects)} transects.")
    return transects

def build_project_paths(city: str):
    """
    Creates and returns all required FEGLA paths in a clean structure:
        root/
        ├── data/<city>/
        ├── outputs/<city>/
            ├── production/
            ├── transects/
            ├── ...
    """

    root = Path.cwd().resolve().parents[0]

    paths = {
        "root"           : root,
        "output_path"    : root / "outputs" / city,
        "production_path": root / "outputs" / city / "production",
        "transect_path"  : root / "outputs" / city / "transects",
        "data_path"      : root / "data"   / city,
    }

    # Create directories that must exist
    paths["output_path"].mkdir(parents=True, exist_ok=True)
    paths["production_path"].mkdir(parents=True, exist_ok=True)
    paths["transect_path"].mkdir(parents=True, exist_ok=True)

    return paths