#--------------------------------------------------------
#
# PACKAGES
#
#--------------------------------------------------------
import os
import sys
from tqdm import tqdm
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.ticker as ticker
from matplotlib.patches import Polygon as MplPolygon
from sklearn.cluster import KMeans
from pyproj import Transformer
import matplotlib.ticker as mticker
import matplotlib.path as mpath
from pathlib import Path
import math
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Polygon, MultiPolygon
import simplekml
from simplekml import Color
from matplotlib.lines import Line2D

# Crear un transformador UNA sola vez
transformer = Transformer.from_crs("epsg:4326", "epsg:32719", always_xy=True)

from tsunamicore.fegla.operations import find_maxhorizontalflood

#--------------------------------------------------------
#--------------------------------------------------------
def load_data_iterations(main_path, file_pattern, name_extraction_fn):
    """
    Loads `height_iteration[-1]` and `R0` from multiple PKL files into a structured dictionary.

    Parameters:
    - main_path (Path): Path object to the directory containing PKL files.
    - file_pattern (str): Pattern to match files (e.g., '*cte*', '*squared*', '*linear*').
    - name_extraction_fn (function): Function to extract keys for dictionary storage.

    Returns:
    - dict: A dictionary structured as {F0_X: {Transect: {'height': ..., 'R0': ...}}}.
    """
    results_list = list(main_path.glob(file_pattern))
    results_dict = {}

    for results in tqdm(results_list, desc=f"Loading {file_pattern} files"):
        name_key = name_extraction_fn(str(results))  # Extract name for dict

        with open(results, 'rb') as f:
            data = pickle.load(f)

        # Extract both height and R0 values
        results_dict[name_key] = {
            key: {
                'height': value['height'] if 'height' in value else None,
                'R0': value['R0'] if 'R0' in value else None
            }
            for key, value in data.items()
        }
        del data  # Free memory

    print(f"Loaded {len(results_dict)} simulations for pattern {file_pattern}\n")
    return results_dict

#--------------------------------------------------------
#--------------------------------------------------------
# Function to extract names for CTE and SQUARED
def extract_cte_squared_name(filepath):
    match = re.search(r'F0_(\d+(?:\.\d+)?)', filepath)
    if match:
        return f"F0_{match.group(1)}"
    else:
        return "F0_UNKNOWN"

#--------------------------------------------------------
#--------------------------------------------------------
# Function to extract names for LINEAR (handles F0 and FR separately)
def extract_linear_name(filepath):
    match = re.search(r'F0_(\d+(?:\.\d+)?)_FR_(\d+(?:\.\d+)?)', filepath)
    if match:
        F0_val = match.group(1)
        FR_val = match.group(2)
        return f"F0_{F0_val}_FR_{FR_val}"
    else:
        return "UNKNOWN"

#--------------------------------------------------------
#--------------------------------------------------------

def compute_error_for_extension(results_dict, scen_tran_all):
    """
    Computes the mean error in flood extension for different scenarios based on the results and simulated data.

    Parameters:
    - results_dict (dict): Dictionary structured as {F0_X: {Scenario: {Transect: height_iteration[-1]}}}.
    - scen_tran_all (dict): Dictionary structured as {Scenario: {Transect: hmax values}}.
    - max_scen (int): The maximum number of scenarios to compute errors for. Default is 50.

    Returns:
    - error_extension (dict): Dictionary structured as {F0_X: {"extent_error": [mean_scenario_errors]}}.
    """

    error_extension = {}

    for f0_key, dictionary in tqdm(results_dict.items(), desc="Computing errors for extensions"):
        scen_list = sorted(set(key.split('_')[0].strip('/') for key in dictionary.keys()))
        mean_scenario_errors = []

        for scen in scen_list:
            # Extract data for this scenario
            results_scen = {k: v for k, v in dictionary.items() if k.startswith(scen)}
            data_scen = {k: v for k, v in scen_tran_all.items() if k.startswith(scen)}

            error_tran = []

            for transect_key in results_scen.keys():
                if transect_key not in data_scen:
                    continue  # Skip missing transects

                h_pred = results_scen[transect_key]['height'] # Predicted height
                h_sim = data_scen[transect_key]  # Simulated height

                if len(h_pred) == 0 or len(h_sim) == 0:
                    continue  # Skip empty values

                try:
                    # Compute percentage flood extent error
                    extent_error = np.abs(len(h_pred) - len(h_sim)) / len(h_sim) * 100
                    
                    error_tran.append(extent_error)
                except Exception:
                    pass  # Ignore errors

            if error_tran:
                mean_scenario_errors.append(np.mean(error_tran))  # Compute mean error for the scenario

        if mean_scenario_errors:
            error_extension[f0_key] =  mean_scenario_errors  # Store all mean errors per F0
        
    return error_extension

#--------------------------------------------------------
#--------------------------------------------------------

from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import math
def set_smart_y_ticks(ax, target_majors=5, symmetric_if_cross_zero=False, step=None):
    """
    Set major/minor yticks with nice limits, without expanding beyond the data range.
    """
    ymin, ymax = ax.get_ylim()
    if ymin == ymax:
        ymin, ymax = ymin - 1, ymax + 1

    if symmetric_if_cross_zero and ymin < 0 < ymax:
        m = max(abs(ymin), abs(ymax))
        ymin, ymax = -m, m

    rng = ymax - ymin
    if step is None:
        raw = rng / max(1, target_majors - 1)
        mag = 10 ** math.floor(math.log10(raw))
        for base in (1, 2, 2.5, 5, 10):
            s = base * mag
            if s >= raw:
                step = s
                break

    # round down min, round up max
    lo = math.floor(ymin / step) * step
    hi = math.ceil(ymax / step) * step

    # apply
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    return step, lo, hi


#--------------------------------------------------------
#--------------------------------------------------------

def plot_1Derror_distribution(
    error_constant, error_squared, outfigPath, Nsim,
    error_type="Area", figsize=(12, 4.5),
    cmap="Spectral",
    clip_percentile=None,   # e.g., 99 to visually clip heavy outliers
    bw=0.3,                 # KDE smoothing for the violin
    bias_constant=None,     # dict like {"F0_0.8":[...], ...} with bias in %
    bias_squared=None       # dict like {"F0_0.8":[...], ...} with bias in %
):
    """
    Split violin per F0 showing the distribution of Absolute Error (EA, %):
    - Left half = constant, Right half = squared
    - Black box = IQR (Q1–Q3)
    - Short black tick = MEAN of EA (not median)
    - Dashed lines = mean EA curves (per mode) + red 'x' at minimal EA per mode
    - Optional: overlay mean BIAS (|bias|) as markers on the same axis:
        '^' if bias ≥ 0 (overestimation), 'v' if bias < 0 (underestimation)

    Parameters
    ----------
    error_constant, error_squared : dict
        {"F0_x": [ea%, ea%, ...], ...} for each mode.
    outfigPath : str or Path
        Folder to save the figure.
    Nsim : int
        Number of simulations used (for title/tagging).
    error_type : str
        Label in the output filename.
    figsize : tuple
        Figure size (width, height) in inches.
    cmap : str
        Colormap name for per-F0 coloring (not crucial here).
    clip_percentile : float or None
        If set, visually trims EA distribution to [100-p, p] percentiles.
    bw : float
        Violin KDE smoothing passed to matplotlib's violinplot.
    bias_constant, bias_squared : dict or None
        If provided, mean bias per F0 is overlaid as markers: '^' (positive), 'v' (negative),
        using the magnitude |bias| on the same EA axis.

    Returns
    -------
    min_F0_const, min_err_const, min_F0_sq, min_err_sq : tuple
        F0 and minimum mean EA (%) for constant and squared, respectively.
    """

    def f0_from_key(k): 
        return float(k.split("_", 1)[1])

    # --- F0 domain (intersection) ---
    f0_c = sorted(f0_from_key(k) for k in error_constant.keys())
    f0_s = sorted(f0_from_key(k) for k in error_squared.keys())
    f0_sorted = sorted(set(f0_c).intersection(f0_s))
    if not f0_sorted:
        raise ValueError("No intersection of F0 between constant and squared.")

    # --- Build long-form DataFrame for errors ---
    rows = []
    for k, vals in error_constant.items():
        f0 = f0_from_key(k)
        if f0 in f0_sorted:
            rows.extend({"F0": f0, "error": float(e), "mode": "constant"} for e in vals)
    for k, vals in error_squared.items():
        f0 = f0_from_key(k)
        if f0 in f0_sorted:
            rows.extend({"F0": f0, "error": float(e), "mode": "squared"} for e in vals)

    df = pd.DataFrame(rows)
    df["F0"] = pd.Categorical(df["F0"], categories=f0_sorted, ordered=True)

    # Optional visual clipping to avoid extremely long violins
    if clip_percentile is not None and len(df):
        hi = np.nanpercentile(df["error"], clip_percentile)
        lo = np.nanpercentile(df["error"], 100 - clip_percentile)
        df = df[(df["error"] >= lo) & (df["error"] <= hi)].copy()

    # Mean EA per mode/F0
    means = (
        df.groupby(["mode", "F0"], observed=True)["error"]
          .mean().reset_index().sort_values(["mode", "F0"])
    )
    def series_mean(m):
        s = means[means["mode"] == m].set_index("F0")["error"]
        return [float(s.get(f, np.nan)) for f in f0_sorted]

    mean_const = series_mean("constant")
    mean_sq    = series_mean("squared")

    # Locate minimal mean EA per mode
    min_idx_c = int(np.nanargmin(mean_const))
    min_idx_s = int(np.nanargmin(mean_sq))
    min_F0_const, min_err_const = f0_sorted[min_idx_c], mean_const[min_idx_c]
    min_F0_sq,   min_err_sq    = f0_sorted[min_idx_s], mean_sq[min_idx_s]

    # Colors per mode (consistent with your palette)
    color_const = "#003f5c"  # blue
    color_sq    = "#ffa600"  # orange

    fig, ax = plt.subplots(figsize=figsize)
    plt.rcParams.update({"font.size": 14})
    x = np.arange(len(f0_sorted), dtype=float)
    width = 0.6  # total violin width

    # --- Half-violin helper (IQR and MEAN tick) ---
    def half_violin(ax, data, pos, facecolor, side="left",
                    width=width, bw=bw, center_overlap=0.02):
        """
        Draw a half violin with IQR (vertical black bar) and MEAN tick (short horizontal line).
        center_overlap slightly overlaps towards the center to avoid a 'hard cut'.
        """
        if len(data) == 0:
            return
        vp = ax.violinplot([data], positions=[pos], widths=width,
                           showmeans=False, showextrema=False, showmedians=False,
                           bw_method=bw)
        for b in vp["bodies"]:
            b.set_facecolor(facecolor)
            b.set_edgecolor("black")
            b.set_linewidth(0.8)
            b.set_alpha(0.95)

            # Edit vertices to keep only half
            verts = b.get_paths()[0].vertices.copy()
            if side == "left":
                verts[:, 0] = np.minimum(verts[:, 0], pos + center_overlap)
            else:
                verts[:, 0] = np.maximum(verts[:, 0], pos - center_overlap)

            if hasattr(b, "set_verts"):
                b.set_verts([verts])
            else:
                b.set_paths([mpath.Path(verts)])

        # IQR (Q1–Q3)
        q1, q3 = np.nanpercentile(data, [25, 75])
        ax.plot([pos, pos], [q1, q3], color="black", linewidth=4, solid_capstyle="butt")

        # MEAN tick (not median)
        mean_val = float(np.nanmean(data))
        ax.plot([pos - 0.06, pos + 0.06], [mean_val, mean_val], color="black", linewidth=2)

    # --- Draw split violins per F0 ---
    for i, f in enumerate(f0_sorted):
        d_c = df[(df["mode"] == "constant") & (df["F0"] == f)]["error"].to_numpy(float)
        d_s = df[(df["mode"] == "squared")  & (df["F0"] == f)]["error"].to_numpy(float)
        if d_c.size: half_violin(ax, d_c, x[i], color_const, side="left")
        if d_s.size: half_violin(ax, d_s, x[i], color_sq,    side="right")

    # --- Mean EA curves + minima marks ---
    ax.plot(x, mean_const, color=color_const, ls="--", label="Mean EA – Constant")
    ax.plot(x, mean_sq,    color=color_sq,    ls="--", label="Mean EA – Squared")
    ax.scatter([x[min_idx_c]], [min_err_const], s=90, marker="x", color="red", zorder=5, label="Min EA")
    ax.scatter([x[min_idx_s]], [min_err_sq],    s=90, marker="x", color="red", zorder=5)

    # --- OPTIONAL: overlay mean bias as |bias| with sign in marker (^ / v) ---
    def _mean_bias_series(bias_dict):
        rows_b = []
        for k, vals in (bias_dict or {}).items():
            f0 = f0_from_key(k)
            if f0 in f0_sorted:
                rows_b.extend({"F0": f0, "bias": float(b)} for b in vals)
        if not rows_b:
            return None
        dfb = pd.DataFrame(rows_b)
        dfb["F0"] = pd.Categorical(dfb["F0"], categories=f0_sorted, ordered=True)
        mean_b = dfb.groupby("F0", observed=True)["bias"].mean()
        return [float(mean_b.get(f, np.nan)) for f in f0_sorted]

    if (bias_constant is not None) and (bias_squared is not None):
        mean_bias_const = _mean_bias_series(bias_constant)
        mean_bias_sq    = _mean_bias_series(bias_squared)

        # plot bias (EB) as dashed lines
        ax.plot(x, mean_bias_const, color=color_const, ls=":", 
                label="Mean EB – Constant")
        ax.plot(x, mean_bias_sq,    color=color_sq,    ls=":", 
                label="Mean EB – Squared")
        
    # --- Locate and mark EB closest to zero (best bias) ---
    if mean_bias_const is not None and mean_bias_sq is not None:
        min_idx_bc = int(np.nanargmin(np.abs(mean_bias_const)))
        min_idx_bs = int(np.nanargmin(np.abs(mean_bias_sq)))
        min_F0_bias_const, best_bias_const = f0_sorted[min_idx_bc], mean_bias_const[min_idx_bc]
        min_F0_bias_sq,   best_bias_sq    = f0_sorted[min_idx_bs], mean_bias_sq[min_idx_bs]

        ax.scatter([x[min_idx_bc]], [best_bias_const], s=90, marker="^", 
                color="m", zorder=6, label="Best EB – Constant")
        ax.scatter([x[min_idx_bs]], [best_bias_sq],    s=90, marker="v", 
                color="m", zorder=6, label="Best EB – Squared")


    # --- Aesthetics & save ---
    ax.set_xlabel("F0")
    ax.set_ylabel("Absolute Error (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in f0_sorted], rotation=0)
    set_smart_y_ticks(ax, target_majors=6)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_title(
        f"Constant: F0 = {min_F0_const} with EA = {min_err_const:.2f}%  |  "
        f"Squared: F0 = {min_F0_sq} with EA = {min_err_sq:.2f}%",
        fontsize=14
    )

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), 
               frameon=True, ncol=4, fontsize=11)

    outfigPath = Path(outfigPath); outfigPath.mkdir(parents=True, exist_ok=True)
    fname = outfigPath / f"{error_type}_error_distribution_with_bias_Nsim_{Nsim}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show(block=False); plt.pause(0.1)

    return min_F0_const, min_err_const, min_F0_sq, min_err_sq

#--------------------------------------------------------
#--------------------------------------------------------

def plot_2Derror_distribution(error_data, outfigPath, Nsim, plot_type,
                              cmap="Blues", target_levels=8, include_min_zero=True,
                              bias_data=None, eb_levels=(-20, -10, 0, 10, 20),  # niveles EXACTOS
                              mark_best_eb=True):
    """
    EA: mapa de contorno relleno (promedio %).
    EB: curvas de nivel firmadas (%, 0 sólido; >0 dashed; <0 dashdot).
    - error_data: {'F0_0.8_FR_0.4': [EA%, ...], ...}
    - bias_data : {'F0_0.8_FR_0.4': [EB%, ...], ...}  [opcional]
    """

    # ---------- Parseo EA a grilla ----------
    rows = []
    for key, vals in error_data.items():
        parts = key.split("_")  # F0_x_FR_y
        F0 = float(parts[1]); FR = float(parts[3])
        rows.append((F0, FR, float(np.mean(vals))))
    if not rows:
        print("No valid error data to plot."); return

    rows = np.array(rows, dtype=float)
    F0_vals, FR_vals, Z_vals = rows[:, 0], rows[:, 1], rows[:, 2]

    F0_u = np.unique(F0_vals)
    FR_u = np.unique(FR_vals)
    F0_grid, FR_grid = np.meshgrid(F0_u, FR_u)  # cols=F0, rows=FR
    Z = np.full_like(F0_grid, np.nan, dtype=float)

    for F0, FR, z in rows:
        i = np.where(FR_u == FR)[0][0]  # fila
        j = np.where(F0_u == F0)[0][0]  # columna
        Z[i, j] = z

    # ---------- (Opcional) Parseo EB a grilla ----------
    B = None
    if bias_data is not None:
        b_rows = []
        for key, vals in bias_data.items():
            parts = key.split("_")
            F0 = float(parts[1]); FR = float(parts[3])
            b_rows.append((F0, FR, float(np.mean(vals))))
        if b_rows:
            b_rows = np.array(b_rows, dtype=float)
            B = np.full_like(F0_grid, np.nan, dtype=float)
            for F0, FR, b in b_rows:
                i = np.where(FR_u == FR)[0][0]
                j = np.where(F0_u == F0)[0][0]
                B[i, j] = b

    # ---------- Niveles "bonitos" para EA ----------
    def auto_levels(vmin, vmax, target=8, nice=(1, 2, 2.5, 5, 10, 20, 25, 50, 100)):
        import math
        if not np.isfinite(vmin): vmin = 0.0
        if not np.isfinite(vmax): vmax = 1.0
        if include_min_zero: vmin = min(0.0, vmin)
        rng = max(vmax - vmin, 1e-9)
        raw = rng / max(target, 1)
        mag = 10 ** math.floor(math.log10(raw))
        candidates = [s * mag for s in nice] + [s * mag * 10 for s in nice]
        step = min(candidates, key=lambda s: abs(s - raw))
        start = step * np.floor(vmin / step)
        end   = step * np.ceil(vmax / step)
        if end <= start:
            end = start + 3 * step
        return np.arange(start, end + 0.5 * step, step)

    vmin = np.nanmin(Z); vmax = np.nanmax(Z)
    levels = auto_levels(vmin, vmax, target=target_levels)

    # ---------- Mínimo global de EA ----------
    min_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
    min_F0 = float(F0_grid[min_idx]); min_FR = float(FR_grid[min_idx]); min_val = float(Z[min_idx])

    # ---------- Plot ----------
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 14})

    # EA
    cf = plt.contourf(F0_grid, FR_grid, Z, levels=levels, cmap=cmap, extend="both")
    cbar = plt.colorbar(cf); cbar.set_label("Error % (EA)", fontsize=12)
    if len(levels) > 12:
        cbar.locator = mticker.MaxNLocator(nbins=10); cbar.update_ticks()
    else:
        cbar.set_ticks(levels)

    cs = plt.contour(F0_grid, FR_grid, Z, levels=levels, colors="black", linewidths=0.55)
    plt.clabel(cs, fmt="%.0f", fontsize=8)

    # Punto de mínimo EA
    plt.scatter(min_F0, min_FR, c="red", marker="x", s=80, label="Min EA")

    # ---------- EB (sesgo) como curvas de nivel firmadas ----------
    legend_handles = []
    if B is not None and np.isfinite(B).any():
        # niveles EXACTOS que se piden, respetando el signo
        eb_levels = tuple(sorted(set(eb_levels)))  # (-10, -5, 0, 5, 10) por defecto

        # estilos: 0 sólido, >0 dashed (subestima), <0 dashdot (sobrestima)
        ls_map = {lv: ("solid" if lv == 0 else "dashed" if lv > 0 else "dashdot")
                  for lv in eb_levels}

        csb = plt.contour(
            F0_grid, FR_grid, B,
            levels=eb_levels,
            colors="m", linewidths=1,
            linestyles=[ls_map[lv] for lv in eb_levels]
        )
        plt.clabel(csb, fmt="%+g%%", fontsize=8)

        # Leyenda EB
        legend_handles.append(Line2D([0],[0], color="m", lw=1.6, ls="solid",  label="EB = 0%"))
        if any(lv > 0 for lv in eb_levels):
            legend_handles.append(Line2D([0],[0], color="m", lw=1.2, ls="dashed",  label="EB > 0 (Underestimate)"))
        if any(lv < 0 for lv in eb_levels):
            legend_handles.append(Line2D([0],[0], color="m", lw=1.2, ls="dashdot", label="EB < 0 (Overestimate)"))

        # Punto con EB más cercano a 0 (desbalance mínimo)
        if mark_best_eb:
            absB = np.abs(B)
            bi, bj = np.unravel_index(np.nanargmin(absB), absB.shape)
            best_F0, best_FR, best_eb = float(F0_grid[bi, bj]), float(FR_grid[bi, bj]), float(B[bi, bj])
            plt.scatter(best_F0, best_FR, c="m", marker="^", s=80,
                        label=f"Best EB ({best_eb:+.2f}%)")

    # ---------- Ejes y grillas ----------
    ax = plt.gca()
    plt.xlabel("Froude number on the Coast (F0)")
    plt.ylabel("Froude number on the Runup (FR)")

    if len(F0_u) > 1:
        step_F0 = float(np.round(np.diff(F0_u).min(), 2))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(step_F0))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(max(step_F0/2, 1e-9)))
    if len(FR_u) > 1:
        step_FR = float(np.round(np.diff(FR_u).min(), 2))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(step_FR))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(max(step_FR/2, 1e-9)))

    ax.grid(which="major", linestyle="--", alpha=0.65)
    ax.grid(which="minor", linestyle="--", alpha=0.35)

    plt.title(f"Min EA at F0={min_F0:.1f}, FR={min_FR:.1f}  (EA={min_val:.2f}%)", fontsize=13)

    # ---------- Leyenda combinada ----------
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(legend_handles)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=2, frameon=True, fontsize=12)

    # ---------- Guardado ----------
    outfigPath = Path(outfigPath); outfigPath.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfigPath / f"{plot_type}error_distribution_Nsim_{Nsim}_linear.png",
                dpi=300, bbox_inches="tight")
    plt.show(block=False); plt.pause(0.1)

    return min_F0, min_FR, float(min_val)

#--------------------------------------------------------
#--------------------------------------------------------
def plot_best_fit_models(error_resultsCTE, error_resultsSquared, error_resultslinear, 
                         resultsCte_dict, min_F0_cte, min_F0_squared, min_F0_linear, min_FR_linear):
    """
    Plots the error analysis for the best-fit models using:
    1. A line plot comparing error evolution across simulations.
    2. A boxplot summarizing error distributions for each model.

    Parameters:
    - error_resultsCTE (dict): Dictionary containing errors for constant Froude models.
    - error_resultsSquared (dict): Dictionary containing errors for squared Froude models.
    - error_resultslinear (dict): Dictionary containing errors for linear Froude models.
    - resultsCte_dict (dict): Dictionary containing simulation results for constant Froude models.
    - min_F0_cte (str): Best-fit Froude number for the constant model.
    - min_F0_squared (str): Best-fit Froude number for the squared model.
    - min_F0_linear (str): Best-fit Froude number for the linear model.
    - min_FR_linear (str): Best-fit FR value for the linear model.

    Returns:
    - None (displays the plot).
    """

    # Create figure with two subplots (Line plot + Boxplot)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [2, 1]})

    # Generate keys for the best-fit models
    min_CteKey = f'F0_{min_F0_cte}'
    min_SquaredKey = f'F0_{min_F0_squared}'
    min_LinearKey = f'F0_{min_F0_linear}_FR_{min_FR_linear}'

    # Extract all unique scenarios from dictionary keys
    Allscenarios = sorted(set(key.split('_')[0].strip('/') for key in resultsCte_dict[min_CteKey].keys()))

    # Extract the best-fit model errors
    errorCte_opti     = error_resultsCTE[min_CteKey]
    errorSquared_opti = error_resultsSquared[min_SquaredKey]
    errorLinear_opti  = error_resultslinear[min_LinearKey]

    # Extract best-fit Froude numbers
    cteKey     = min_CteKey.split('_')[1]
    SquaredKey = min_SquaredKey.split('_')[1]
    LinearKey  = min_LinearKey.split('_')[1], min_LinearKey.split('_')[3]

    # Format best-fit model names for legends
    legend_cte     = f'Cte F0 = {cteKey}'
    legend_squared = f'Squared F0 = {SquaredKey}'
    legend_linear  = f'Linear F0={LinearKey[0]} and FR={LinearKey[1]}'

    # Compute mean and standard deviation for squared and linear models
    mean_squared = np.mean(np.abs(errorSquared_opti))
    std_squared = np.std(np.abs(errorSquared_opti))
    mean_linear = np.mean(np.abs(errorLinear_opti))
    std_linear = np.std(np.abs(errorLinear_opti))

    # Define consistent colors for the plots
    colors = ['#1f77b4', '#ff7f0e', 'm']

    # === Line Plot (Evolution of EA % across simulations) ===
    ax[0].plot(np.linspace(1, len(Allscenarios), len(Allscenarios)), errorCte_opti, marker='x', ls='--', lw=0.5, 
               label=legend_cte, color=colors[0])
    ax[0].plot(np.linspace(1, len(Allscenarios), len(Allscenarios)), errorSquared_opti, marker='o', ls='--', lw=0.5, 
               label=legend_squared, color=colors[1])
    ax[0].plot(np.linspace(1, len(Allscenarios), len(Allscenarios)), errorLinear_opti, marker='v', ls='--', lw=0.5, 
               label=legend_linear, color=colors[2])
    ax[0].legend()
    ax[0].grid(True, linestyle='--', alpha=0.5)
    ax[0].set_xlabel("Simulations", fontsize=12)
    ax[0].set_ylabel("EA %", fontsize=12)
    ax[0].set_xticks(np.arange(2, len(Allscenarios) + 1, 4))  # Set ticks every 4 simulations
    ax[0].set_xlim(0, len(Allscenarios) + 0.5)

    # === Boxplot (Distribution of Errors) ===
    # Create a DataFrame for plotting
    data = pd.DataFrame({
        'Error': list(np.abs(errorCte_opti)) + list(np.abs(errorSquared_opti)) + list(np.abs(errorLinear_opti)),
        'Category': ['Constant'] * len(errorCte_opti) + 
                    ['Squared'] * len(errorSquared_opti) + 
                    ['Linear'] * len(errorLinear_opti)
    })

    # Plot boxplot with category colors
    sns.boxplot(
        x='Category', 
        y='Error', 
        data=data, 
        ax=ax[1], 
        hue='Category',  # Assign hue to match categories
        palette=colors,
        legend=False
    )
    ax[1].set_ylabel("EA(%)", fontsize=12)
    ax[1].grid(True, linestyle='--', alpha=0.7)
    ax[1].set_xlabel('Froude Number')

    # Add statistics text to the boxplot
    stats_text = (
        f"Squared:  Mean: {mean_squared:.2f}  Std: {std_squared:.2f}\n"
        f"Linear:  Mean: {mean_linear:.2f}  Std: {std_linear:.2f}"
    )
    ax[1].text(0.25, 0.9, stats_text, transform=ax[1].transAxes, fontsize=10,
               verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()
    
#--------------------------------------------------------
#--------------------------------------------------------

def integrate_hmax_predictions(transectData, resultsCte_dict, resultsSquared_dict, resultsLinear_dict,
                               min_F0_cte, min_error_cte, 
                               min_F0_squared, min_error_squared, 
                               min_F0_linear, min_FR_linear, min_error_linear):
    """
    Integrates only the hmax predictions corresponding to the model with the minimum error into the SWE simulation dataset.

    Parameters:
    - transectData (dict): Dictionary containing DataFrames of SWE simulations, with keys as transect IDs.
    - resultsCte_dict (dict): Dictionary containing hmax predictions from the constant Froude model.
    - resultsSquared_dict (dict): Dictionary containing hmax predictions from the squared Froude model.
    - resultsLinear_dict (dict): Dictionary containing hmax predictions from the linear Froude model.
    - min_F0_cte (str): Best-fit Froude number for the constant model.
    - min_error_cte (float): Minimum error for the constant model.
    - min_F0_squared (str): Best-fit Froude number for the squared model.
    - min_error_squared (float): Minimum error for the squared model.
    - min_F0_linear (str): Best-fit Froude number for the linear model.
    - min_FR_linear (str): Best-fit FR value for the linear model.
    - min_error_linear (float): Minimum error for the linear model.

    Returns:
    - updated_transectData (dict): A new dictionary with an integrated `hmax_best` column containing the best hmax predictions.
    """
    # Helper function to format floats without trailing .0
    def format_froude(value):
        """Sanitize Froude numbers to avoid '.0' in keys like F0_0.0."""
        val = float(value)  # asegura conversión
        return str(int(val)) if val.is_integer() else str(val)

    # Format all keys
    key_cte = f'F0_{format_froude(min_F0_cte)}'
    key_squared = f'F0_{format_froude(min_F0_squared)}'
    key_linear = f'F0_{format_froude(min_F0_linear)}_FR_{format_froude(min_FR_linear)}'

    # Determine which model has the lowest error
    error_mapping = {
        min_error_cte: ('cte', resultsCte_dict.get(key_cte, {}), key_cte),
        min_error_squared: ('squared', resultsSquared_dict.get(key_squared, {}), key_squared),
        min_error_linear: ('linear', resultsLinear_dict.get(key_linear, {}), key_linear)
    }

    # Identify the best-fit model
    min_error_value = min(error_mapping.keys())
    best_model_name, best_model_data, best_model_key = error_mapping[min_error_value]

    # Print which model was chosen
    print(f"\nIntegrating hmax from the best-fit model: {best_model_name} (Error: {min_error_value:.2f})")

    # Create a copy of transectData to store the updates
    updated_transectData = {}

    # Iterate over each transect in transectData
    for transect_id, df in transectData.items():
        # Copy the DataFrame to avoid modifying the original
        df = df.copy()

        hmax_best = best_model_data.get(transect_id, {}).get('height', np.full(len(df), np.nan))

        # Convert to array and flatten in case it's 2D (e.g., (1, N))
        hmax_best = np.asarray(hmax_best).flatten()

        # If all values are nan or empty, just fill with NaNs
        if np.isnan(hmax_best).all() or hmax_best.size == 0:
            df['hmax_best'] = np.full(len(df), np.nan)
        else:
            df['hmax_best'] = np.pad(hmax_best, (0, max(0, len(df) - len(hmax_best))), constant_values=np.nan)[:len(df)]

        # Store the updated DataFrame
        updated_transectData[transect_id] = df

    return updated_transectData, best_model_name, best_model_key

#--------------------------------------------------------
#--------------------------------------------------------

def calculate_polygon_and_area(first_points, last_points):
    """
    Optimized version:
    - Avoids GeoDataFrame and repeated to_crs transformations.
    - Uses pyproj Transformer directly for fast coordinate projection.
    """
    # Combine points into one array
    points = np.vstack([first_points, last_points[::-1]])
    
    # Transform coordinates to UTM
    x, y = transformer.transform(points[:, 0], points[:, 1])
    
    # Create polygon in projected coordinates
    polygon_proj = Polygon(np.column_stack((x, y)))
    
    # Calculate area in square meters directly
    area_m2 = polygon_proj.area
    
    return polygon_proj, area_m2

#--------------------------------------------------------
#--------------------------------------------------------
def get_boundary_points(data, column):
    """
    Extracts the boundary points (first and last) for each transect in the provided data,
    calculates a polygon that connects these boundary points, and computes the polygon's area.

    Parameters:
    - data (dict): A dictionary where keys are transect IDs, and values are DataFrames containing
      'lon', 'lat', and a specified column (e.g., 'hmax', 'hmax_cte', etc.).
    - column (str): The column name representing water height (e.g., 'hmax', 'hmax_cte').

    Returns:
    - polygonSim (shapely.geometry.Polygon): The polygon created by connecting the boundary points.
    - aream2Sim (float): The area of the polygon in square meters.
    """

    firstptoList = []
    lastptoList = []

    for key, df in data.items():
        # Drop NaN values for the given column
        df = df.dropna(subset=[column])
        
        # Ensure there's data left
        if df.empty or len(df) < 2:
            #print(f"Skipping {key}: Not enough valid points in {column}.")
            continue  # Skip this transect

        try:
            # Extract first and last valid points
            firstptoList.append((df['lon'].values[0], df['lat'].values[0]))  # First point
            lastptoList.append((df['lon'].values[-1], df['lat'].values[-1]))  # Last point
        except Exception as e:
            print(f"Error processing {key}: {e}")
            continue  # Skip if there's an unexpected issue

    if len(firstptoList) == 0 or len(lastptoList) == 0:
        print(f"Warning: No valid points found for column {column}.")
        return None, None  # Return None if no valid data is found

    # Convert lists to NumPy arrays
    firstptoArray = np.array(firstptoList)
    lastptoArray = np.array(lastptoList)

    # Create a polygon and compute the area
    polygonSim, aream2Sim = calculate_polygon_and_area(firstptoArray, lastptoArray)

    return polygonSim, aream2Sim

#--------------------------------------------------------
#--------------------------------------------------------
def plot_topobathymetry_and_contours(x, y, z, elev_min=-90, elev_max=240, elev_delta=30, z0_contour=None, cmap='viridis', ax=None, alpha=1, show_colorbar=False):
    """
    Plot a predefined elevation map with contours and a color bar. A contour for z=0 can also be plotted if provided.

    Parameters:
    - x, y (np.array): 2D arrays of longitude and latitude values for each grid point.
    - z (np.array): 2D array of elevation data at the grid points defined by x and y.
    - elev_min (int): Minimum elevation value for contour levels.
    - elev_max (int): Maximum elevation value for contour levels.
    - elev_delta (int): Interval between contour levels.
    - z0_contour (np.array): Optional. 2D array of x, y coordinates for the z=0 contour line.
    - cmap (str): Colormap for the plot.
    - show_colorbar (bool): If True, display the color bar.

    Returns:
    - ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib subplot axis for further customization.
    """
    # Initialize the plot and set its size
    create_new_fig = ax is None
    if create_new_fig:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    # Define the contour levels
    contour_levels = np.arange(elev_min, elev_max + elev_delta, elev_delta)

    # Fill the contours with color
    cp = ax.contourf(x, y, z, levels=contour_levels, cmap=cmap, alpha=alpha)

    # Draw contour lines
    cs = ax.contour(x, y, z, levels=contour_levels, colors='black', linestyles='solid', linewidths=0.5)
    plt.clabel(cs, inline=True, fontsize=8, fmt='%1.0f m')

    # Show color bar only in the last column
    if show_colorbar:
        cbar = plt.colorbar(cp, ax=ax, label='Elevation (m)', ticks=contour_levels)
        cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(elev_delta))

    # Plot shoreline
    if z0_contour is not None:
        ax.plot(z0_contour[:, 0], z0_contour[:, 1], c='r',ls='--', lw=0.8, label='Shoreline [0 m]')
        #ax.legend()

    return ax

#--------------------------------------------------------
#--------------------------------------------------------
def get_elevation_parameters(bathy):
    """
    Computes the elevation parameters for plotting topobathymetry contours.

    Parameters:
    - bathy (numpy array): Bathymetry data.

    Returns:
    - elev_min (int): Rounded minimum elevation.
    - elev_max (int): Rounded maximum elevation.
    - elev_delta (int): Recommended contour interval.
    """
    elev_min = int(np.floor(np.nanmin(bathy) / 10) * 10)  # Round down to nearest 10
    elev_max = int(np.ceil(np.nanmax(bathy) / 10) * 10)   # Round up to nearest 10

    range_elev = elev_max - elev_min

    # Determine the elevation interval
    if range_elev < 100:
        elev_delta = 10
    elif range_elev < 300:
        elev_delta = 20
    else:
        elev_delta = 30

    return elev_min, elev_max, elev_delta

#--------------------------------------------------------
#--------------------------------------------------------
def compute_flood_extent_areas(updated_transectData):
    """
    Computes the flood extent polygons and areas for all scenarios based on hmax (SWE Simulations) 
    and hmax_best (Best-Fit Model: cte, linear, or squared).
    
    Parameters:
    - updated_transectData (dict): Dictionary containing DataFrames with hmax (SWE) and hmax_best.

    Returns:
    - flood_extent_dict (dict): Dictionary structured as:
      {
          scenario: {
              "swe": (polygon, area),
              "best_fit": (polygon, area)
          }
      }
    """
    
    flood_extent_dict = {}

    # Get all unique scenarios from keys
    unique_scenarios = sorted(set(key.split('_')[0] for key in updated_transectData.keys()))

    for scenario in tqdm(unique_scenarios, desc="Computing flood extent areas"):
        # Extract transects for this scenario
        scenario_transects = {key: value for key, value in updated_transectData.items() if key.startswith(scenario)}

        if not scenario_transects:
            continue

        # Compute boundary polygons and areas
        polygon_swe, area_swe = get_boundary_points(
            {k: v[['lon', 'lat', 'hmax']].dropna() for k, v in scenario_transects.items()},
            column='hmax'
        )
        
        polygon_best, area_best = get_boundary_points(
            {k: v[['lon', 'lat', 'hmax_best']].dropna() for k, v in scenario_transects.items()},
            column='hmax_best'
        )


        # Store results in dictionary
        flood_extent_dict[scenario] = {
            "swe": (polygon_swe, area_swe / 1e6),   # Convert to km²
            "best_fit": (polygon_best, area_best / 1e6)  # Convert to km²
        }

    return flood_extent_dict

#--------------------------------------------------------
#--------------------------------------------------------
def plot_flood_extent_contours(flood_extent_dict, scenario, grid_lon, grid_lat, bathy, shoreline, ax=None, show_legend=False, show_colorbar=False):
    """
    Plots the flood extent boundaries for a given scenario using precomputed flood extent areas.

    Parameters:
    - flood_extent_dict (dict): Dictionary containing polygons and areas for SWE and best-fit models.
    - scenario_index (int): Index of the scenario to plot (used to select from scenario_keys).
    - scenario_keys (list): List of available scenario keys (e.g., ['S0023', 'S0304', ...]).
    - grid_lon, grid_lat (numpy arrays): Longitude and latitude grid for topography.
    - bathy (numpy array): Bathymetry data.
    - shoreline (float): Contour level for shoreline representation.
    - ax (matplotlib axis, optional): If provided, the plot will be drawn on this axis (for subplots).
    - show_legend (bool): If True, display the legend (only for the first subplot).
    - show_colorbar (bool): If True, display the color bar (only for the last column).

    Returns:
    - None (displays a plot).
    """

    
    # Retrieve the precomputed polygons and areas
    polygon_swe, area_swe = flood_extent_dict[scenario]["swe"]
    polygon_best, area_best = flood_extent_dict[scenario]["best_fit"]

    # If no axis is provided, create a new figure
    is_standalone = ax is None
    if is_standalone:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute elevation parameters automatically
    elev_min, elev_max, elev_delta = get_elevation_parameters(bathy)

    # Plot topobathymetry and shoreline
    plot_topobathymetry_and_contours(grid_lon, grid_lat, bathy,
                                     elev_min=elev_min,
                                     elev_max=elev_max,
                                     elev_delta=elev_delta,
                                     z0_contour=shoreline,
                                     cmap='gray',
                                     ax=ax,
                                     alpha=0.6,
                                     show_colorbar=show_colorbar)

    # Plot flood polygons
    ax.add_patch(MplPolygon(np.c_[polygon_swe.exterior.xy[0], polygon_swe.exterior.xy[1]], closed=True, 
                            edgecolor='#ffa600', fill=False, linewidth=2, label='SWE Simulation'))
    
    ax.add_patch(MplPolygon(np.c_[polygon_best.exterior.xy[0], polygon_best.exterior.xy[1]], closed=True, 
                            edgecolor='#003f5c', fill=False, linewidth=2, label='Best-Fit Model'))

    # Set subplot title with computed areas
    ax.set_title(f"Scenario {scenario}\nSWE: {area_swe:.2f} km² | Best-Fit: {area_best:.2f} km²",
                 fontsize=10, pad=12)
    ax.set_aspect('equal')
    # Add legend only in the first subplot
    if show_legend:
        ax.legend(loc="best", fontsize=8, facecolor='white', framealpha=0.8, edgecolor='black')

    # If it's a standalone figure, show the full plot
    if is_standalone:
        fig.suptitle(f"Flood Extent for Scenario {scenario}", fontsize=14, ha='center', x=0.55)
        plt.tight_layout()
        plt.show()

#--------------------------------------------------------
#--------------------------------------------------------
def plot_best_fit_models(error_resultsCTE, error_resultsSquared, error_resultslinear, 
                         resultsCte_dict, min_F0_cte, min_F0_squared, min_F0_linear, min_FR_linear,
                         outfigPath, Nsim, type, n_clusters=20):
    """
    Plots the error analysis for the best-fit models using:
    1. A line plot comparing error evolution across simulations.
    2. A boxplot summarizing error distributions for each model.
    3. Selects the most interesting scenarios using K-means clustering.

    Parameters:
    - error_resultsCTE (dict): Dictionary containing errors for constant Froude models.
    - error_resultsSquared (dict): Dictionary containing errors for squared Froude models.
    - error_resultslinear (dict): Dictionary containing errors for linear Froude models.
    - resultsCte_dict (dict): Dictionary containing simulation results for constant Froude models.
    - min_F0_cte (str): Best-fit Froude number for the constant model.
    - min_F0_squared (str): Best-fit Froude number for the squared model.
    - min_F0_linear (str): Best-fit Froude number for the linear model.
    - min_FR_linear (str): Best-fit FR value for the linear model.
    - Nsim (int): Number of selected scenarios for calibration.
    - n_clusters (int): Number of clusters to select the most representative scenarios.

    Returns:
    - selected_scenarios (list): Indices of the most representative scenarios.
    """
    # Generate keys for the best-fit models
    # Helper function to format floats without trailing .0
    def format_froude(value):
        """
        Formatea el valor de Froude para claves tipo 'F0_x.y' con:
        - 'F0_0' si el valor es 0
        - 'F0_x.y' con 1 decimal en caso contrario
        """
        val = float(value)
        if np.isclose(val, 0.0):
            return "0"
        else:
            return f"{val:.1f}"

    # Format all keys
    min_CteKey = f'F0_{format_froude(min_F0_cte)}'
    min_SquaredKey = f'F0_{format_froude(min_F0_squared)}'
    min_LinearKey = f'F0_{format_froude(min_F0_linear)}_FR_{format_froude(min_FR_linear)}'

    # Extract all unique scenarios from dictionary keys
    Allscenarios = sorted(set(key.split('_')[0].strip('/') for key in resultsCte_dict[min_CteKey].keys()))

    # Extract the best-fit model errors
    errorCte_opti     = error_resultsCTE[min_CteKey]
    errorSquared_opti = error_resultsSquared[min_SquaredKey]
    errorLinear_opti  = error_resultslinear[min_LinearKey]

    # Stack errors for clustering
    error_matrix = np.vstack([errorCte_opti, errorSquared_opti, errorLinear_opti]).T

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(error_matrix)
    cluster_labels = kmeans.labels_
    selected_scenarios = []
    
    # Select one scenario per cluster (closest to cluster center)
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_center = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(error_matrix[cluster_indices] - cluster_center, axis=1)
        best_index = cluster_indices[np.argmin(distances)]
        selected_scenarios.append(Allscenarios[best_index])

    # Create figure with two subplots (Line plot + Boxplot)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [2, 1]})

    # Extract best-fit Froude numbers
    cteKey     = min_CteKey.split('_')[1]
    SquaredKey = min_SquaredKey.split('_')[1]
    LinearKey  = min_LinearKey.split('_')[1], min_LinearKey.split('_')[3]

    # Format best-fit model names for legends
    legend_cte     = f'Cte F0 = {cteKey}'
    legend_squared = f'Squared F0 = {SquaredKey}'
    legend_linear  = f'Linear F0={LinearKey[0]} and FR={LinearKey[1]}'

    # Compute mean and standard deviation for squared and linear models
    mean_squared = np.mean(np.abs(errorSquared_opti))
    std_squared = np.std(np.abs(errorSquared_opti))
    mean_linear = np.mean(np.abs(errorLinear_opti))
    std_linear = np.std(np.abs(errorLinear_opti))

    # Define consistent colors for the plots
    colors = ['#1f77b4', '#ff7f0e', 'm']

    # === Line Plot ===
    ax[0].plot(np.linspace(1, len(Allscenarios), len(Allscenarios)), errorCte_opti, marker='x', ls='--', lw=0.5, 
               label=legend_cte, color=colors[0])
    ax[0].plot(np.linspace(1, len(Allscenarios), len(Allscenarios)), errorSquared_opti, marker='o', ls='--', lw=0.5, 
               label=legend_squared, color=colors[1])
    ax[0].plot(np.linspace(1, len(Allscenarios), len(Allscenarios)), errorLinear_opti, marker='v', ls='--', lw=0.5, 
               label=legend_linear, color=colors[2])

    ax[0].legend()
    ax[0].grid(True, linestyle='--', alpha=0.5)
    ax[0].set_xlabel("Simulations", fontsize=12)
    ax[0].set_ylabel("EA %", fontsize=12)
    ax[0].set_xticks(np.arange(2, len(Allscenarios) + 1, 4))  # Set ticks every 4 simulations
    ax[0].set_xlim(0, len(Allscenarios) + 0.5)

    # === Boxplot ===
    data = pd.DataFrame({
        'Error': list(np.abs(errorCte_opti)) + list(np.abs(errorSquared_opti)) + list(np.abs(errorLinear_opti)),
        'Category': ['Constant'] * len(errorCte_opti) + 
                    ['Squared'] * len(errorSquared_opti) + 
                    ['Linear'] * len(errorLinear_opti)
    })

    sns.boxplot(x='Category', y='Error', data=data, ax=ax[1], hue='Category', palette=colors, legend=False)
    ax[1].set_ylabel("EA(%)", fontsize=12)
    ax[1].grid(True, linestyle='--', alpha=0.7)
    ax[1].set_xlabel('Froude Number')

    stats_text = (
        f"Squared:  Mean: {mean_squared:.2f}  Std: {std_squared:.2f}\n"
        f"Linear:  Mean: {mean_linear:.2f}  Std: {std_linear:.2f}"
    )
    ax[1].text(0.25, 0.9, stats_text, transform=ax[1].transAxes, fontsize=10,
               verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(outfigPath / f"Best-fit_models_along_scenarios_Nsim_{Nsim}_{type}.png", dpi=300, bbox_inches='tight', format='png')
    plt.show(block=False)
    plt.pause(0.1)

    return selected_scenarios

#--------------------------------------------------------
#--------------------------------------------------------

def plot_froude_parameterization(min_F0_cte, min_F0_squared, min_F0_linear, min_FR_linear, outfigPath, Nsim, type):
    """
    Plots Froude parameterization for constant, linear, and squared models.
    
    Parameters:
    - min_F0_cte (float): Constant Froude number.
    - min_F0_squared (float): Initial Froude number for squared model.
    - min_F0_linear (float): Initial Froude number for linear model.
    - min_FR_linear (float): Final Froude number for linear model.
    
    Returns:
    - None (displays the plot).
    """

    # Ensure values are floats
    min_F0_cte = float(min_F0_cte)
    min_F0_squared = float(min_F0_squared)
    min_F0_linear = float(min_F0_linear)
    min_FR_linear = float(min_FR_linear)

    # Generate distance vector
    distance = np.linspace(0, 1, 100)  # Normalized distance from 0 to 1
    XR = distance[-1]  # Last point of the distance vector

    # Compute Froude parameterizations
    froude_cte = np.full_like(distance, min_F0_cte)  # Constant value
    froude_linear = min_F0_linear + (min_FR_linear - min_F0_linear) * (distance / XR)
    froude_squared = min_F0_squared * (1 - distance / XR) ** 0.5

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(distance, froude_cte, label=f"Constant F0 = {min_F0_cte}", linestyle="--")
    plt.plot(distance, froude_linear, label=f"Linear: F0={min_F0_linear}, FR={min_FR_linear}", linestyle="-.")
    plt.plot(distance, froude_squared, label=f"Squared: F0={min_F0_squared}", linestyle=":")

    # Labels and legend
    plt.xlabel("Normalized Distance")
    plt.ylabel("Froude Number")
    plt.title("Froude Parameterization")
    plt.legend()
    plt.grid(True)
    plt.savefig(outfigPath / f"Best-fit_froude_parameterization_Nsim_{Nsim}_{type}.png", dpi=300, bbox_inches='tight', format='png')
    plt.show(block=False)
    plt.pause(0.01)
#--------------------------------------------------------
#--------------------------------------------------------

def initial_heights_by_scenario(data_dict):
    """
    This function calculates the min, max, mean, and std of the initial height (df['height'].iloc[0]) 
    for all transects 'T00X' within each 'S00X'.

    Parameters:
    data_dict (dict): Dictionary where the keys are 'S00X_T00X' and the values are DataFrames with a 'height' column.

    Returns:
    dict: A dictionary where each key is 'S00X' and the value is a list with [min, max, mean, std] 
          of the initial heights for all 'T00X' transects within that 'S00X'.
    """
    result_dict = {}

    # Iterate through the dictionary and collect initial heights for each 'S00X'
    for key, df in tqdm(data_dict.items()):
        s_key = key.split('_')[0]  # Extract the 'S00X' part of the key
        initial_height = df['hmax'].iloc[0]  # Get the first value of the 'height' column
        
        # If 'S00X' is not already in the result_dict, initialize an empty list
        if s_key not in result_dict:
            result_dict[s_key] = []

        # Append the initial height to the list for the corresponding 'S00X'
        result_dict[s_key].append(initial_height)

    # Now calculate min, max, mean, and std for each 'S00X'
    summary_dict = {}
    for s_key, heights in result_dict.items():
        # Convert the list of heights to a NumPy array for easier calculation
        heights_array = np.array(heights)
        summary_dict[s_key] = [np.nanmin(heights_array), np.nanmax(heights_array), np.nanmean(heights_array), np.nanstd(heights_array)]

    return summary_dict

#--------------------------------------------------------
#--------------------------------------------------------
def plot_flooding_curve(flood_extent_dict, Hmean, bathtub_area, outfigPath, type):
    """
    Plots the relationship between average shoreline flood depth (Hmean) and flood extent area 
    for the best-fit model (Squared) and SWE simulation.

    Parameters:
    - flood_extent_dict (dict): Dictionary containing flood extent polygons and area values 
                                for different scenarios. Structured as:
                                {scenario: {"swe": [polygon, area], "best_fit": [polygon, area]}}
    - Hmean (list or np.array): List or array of average shoreline flood depths.

    Returns:
    - None (Displays a figure with a scatter plot, top histogram, and side histogram).
    """
    
    # Extract SWE and best-fit (Squared) areas
    AreaSim   = np.array([flood_extent_dict[scenario]["swe"][1] for scenario in flood_extent_dict.keys()]) 
    AreaFEGLA = np.array([flood_extent_dict[scenario]["best_fit"][1] for scenario in flood_extent_dict.keys()])
    index = np.linspace(0, len(Hmean) - 1, 10, dtype=int)
    Hmeansorted = np.array(np.sort(Hmean))
    Areabath    = np.array(list(bathtub_area.values()))
    Areabath = np.sort(Areabath)

    # Create the figure and grid layout
    fig = plt.figure(figsize=(11, 8))
    grid = plt.GridSpec(4, 4, hspace=0.15, wspace=0.15)

    # Central plot (Scatterplot)
    main_ax = fig.add_subplot(grid[1:4, 0:3])
    sns.scatterplot(x=Hmean, y=AreaFEGLA, ax=main_ax, s=20, c='#003f5c', edgecolor=None, label='FEGLA')
    sns.scatterplot(x=Hmean, y=AreaSim, ax=main_ax, s=20, c='#ffa600', edgecolor=None, label='SWE Simulations')
    main_ax.plot(Hmeansorted[index], Areabath[index], c='m',lw=1.5, ls='--', label='Bathtub')
    main_ax.set_xlabel("Average shoreline flood depth [m]", fontsize=12)
    main_ax.set_ylabel("Area [$km^2$]", fontsize=12)
    main_ax.grid(True, linestyle='--', alpha=0.7)
    main_ax.set_xticks(np.arange(0, max(Hmean) + 1, 2))  # Adjust `max_x_value` as needed
    main_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))  # Ensure only integers appear
    main_ax.legend()

    # Top histogram (Distribution of Hmean)
    top_ax = fig.add_subplot(grid[0, 0:3], sharex=main_ax)
    sns.histplot(Hmean, bins=30, kde=True, stat='density', color="gray", alpha=0.5, ax=top_ax)
    top_ax.set_ylabel("Density", fontsize=12)
    top_ax.set_xlabel("")
    top_ax.tick_params(labelbottom=False)

    # Side histogram (Distribution of Areas)
    side_ax = fig.add_subplot(grid[1:4, 3], sharey=main_ax)

    # Define histogram bins
    max_area = np.max([np.max(AreaSim), np.max(AreaFEGLA)])
    bins = np.linspace(0, max_area, 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_spacing = np.mean(np.diff(bin_centers))  # espacio entre barras
    bar_width = bin_spacing * 0.4  # 40% del espacio total

    # Calcular histogramas
    hist_FEGLA, _ = np.histogram(AreaFEGLA, bins=bins, density=True)
    hist_Sim, _ = np.histogram(AreaSim, bins=bins, density=True)

    # Plot side-by-side bars
    side_ax.barh(bin_centers - bar_width / 2, hist_FEGLA, height=bar_width, color='#003f5c', label='FEGLA')
    side_ax.barh(bin_centers + bar_width / 2, hist_Sim, height=bar_width, color='#ffa600', label='SWE Simulations')
    side_ax.tick_params(labelleft=False)

    # Labels and legend
    side_ax.set_xlabel("Density", fontsize=12)
    side_ax.set_ylabel("", fontsize=12)
    side_ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.04, 1.15), ncol=1)
    plt.savefig(outfigPath / f"Flooding_Curve_{type}.png", dpi=300, bbox_inches='tight', format='png')
    plt.show(block=False)
    plt.pause(0.1)

#--------------------------------------------------------
#--------------------------------------------------------
def compute_bathtub_areas(transectData, results_dict, best_model_key):

    # Store areas
    bathtub_area = {}

    # Selecting unique scenarios
    unique_scenarios = sorted(set(key.split('_')[0] for key in results_dict[best_model_key].keys()))

    for scenario in unique_scenarios:

        scenario_transects = {key: value for key, value in results_dict[best_model_key].items() if key.startswith(scenario)}

        bathtubFirstpto = list()
        bathtubLastpto  = list()

        for Skey in scenario_transects.keys():
            flood_transect = transectData[Skey]
            flood_transect.rename(columns=str.lower, inplace=True)
            flood_transect['cum_distance'] = flood_transect.index

            # Load shoreline height from flooded transect
            H0             = flood_transect['hmax'].iloc[0]
            
            # Compute XR for a given R0
            if ~np.isnan(H0):
                X_max, _ = find_maxhorizontalflood(flood_transect, H0)
                ix       = flood_transect[flood_transect['cum_distance'] == X_max].index[0]
                dfbath   = flood_transect.iloc[:int(ix)].copy()
                try:
                    bathtubFirstpto.append((dfbath['lon'].values[0], dfbath['lat'].values[0]))
                except:
                    pass
                try:
                    bathtubLastpto.append((dfbath['lon'].values[-1], dfbath['lat'].values[-1]))
                except:
                    pass
        # Compute area
        _, aream2bath          = calculate_polygon_and_area(bathtubFirstpto, bathtubLastpto)
        bathtub_area[scenario] = aream2bath/10**6

    return bathtub_area

#--------------------------------------------------------
#--------------------------------------------------------
def plot_all_flood_extent_images(
    flood_extent_dict, selected_scenarios, grid_lon, grid_lat, bathy, shoreline, outfigPath, Nsim
):
    """
    Generates one plot per scenario with flood extent contours, saving each as a PNG.

    Parameters:
    - flood_extent_dict (dict): Flood extent polygons and data for different scenarios.
    - selected_scenarios (list): List of scenario keys to plot (e.g., ['S0001', 'S0023', ...]).
    - grid_lon, grid_lat (2D array): Meshgrid longitude and latitude values.
    - bathy (2D array): Bathymetry matrix.
    - shoreline (array): Coordinates of the 0m contour (shoreline).
    - outfigPath (Path): Directory to save individual figures.

    Returns:
    - None (saves individual PNG images for each scenario).
    """

    for i, scenario in enumerate(selected_scenarios):
        # Create a standalone figure
        fig, ax = plt.subplots(figsize=(8, 7))

        # Plot flood extent with standalone = True behavior (no need to trigger suptitle manually)
        plot_flood_extent_contours(
            flood_extent_dict   = flood_extent_dict,
            scenario            = scenario,
            grid_lon            = grid_lon,
            grid_lat            = grid_lat,
            bathy               = bathy,
            shoreline           = shoreline,
            ax                  = ax,
            show_legend         = True,
            show_colorbar       = True
        )
        ax.set_xlabel('Longitude [°]')
        ax.set_ylabel('Latitude [°]')

        # Improve layout and save
        plt.tight_layout()
        fig.savefig(outfigPath / f"FloodExtent_Nsim_{Nsim}_{scenario}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)  # Avoid memory buildup


#--------------------------------------------------------
#--------------------------------------------------------
def compute_all_flood_areas(results_dict, transectData, colname='hmax_FEGLA'):
    """
    Computes shoreline flood areas for each Froude value in the results dictionary
    using both SWE and FEGLA (or custom) modeled wave heights.

    Parameters:
    -----------
    results_dict : dict
        Dictionary structured as results[Fr][simulation] with 'height' key.
    transectData : dict
        Dictionary of transect DataFrames for each simulation.
    colname : str
        Column name to insert modeled heights (default = 'hmax_FEGLA')

    Returns:
    --------
    all_results : dict
        Nested dictionary with structure:
        all_results[Fr] = {
            scenario: {
                'SWE_area': ...,
                'SWE_polygon': ...,
                'FEGLA_area': ...,
                'FEGLA_polygon': ...
            }, ...
        }
    """
    import numpy as np
    from tqdm import tqdm

    all_results = {}

    for Fr_key in results_dict.keys():
        resultsFr = results_dict[Fr_key]
        transectData_dummy = transectData.copy()

        # Agregar columna colname a transectos
        for sim_key in transectData_dummy.keys():
            df = transectData_dummy[sim_key]
            col2add = np.asarray(resultsFr[sim_key]['height']).flatten()
            col2add_ext = np.full(len(df), np.nan)

            minlen = min(len(df), len(col2add))
            col2add_ext[:minlen] = col2add[:minlen]

            df[colname] = col2add_ext
            transectData_dummy[sim_key] = df

        # Agrupar por escenario base (antes del primer "_")
        unique_scenarios = sorted(set(key.split('_')[0] for key in transectData_dummy.keys()))
        area_results = {}

        for scenario in tqdm(unique_scenarios, desc=f"Computing areas for Fr={Fr_key}"):
            scenario_transects = {key: value for key, value in transectData_dummy.items() if key.startswith(scenario)}

            polygon_swe, area_swe = get_boundary_points(
                {k: v[['lon', 'lat', 'hmax']].dropna() for k, v in scenario_transects.items()},
                column='hmax'
            )

            polygon_model, area_model = get_boundary_points(
                {k: v[['lon', 'lat', colname]].dropna() for k, v in scenario_transects.items()},
                column=colname
            )

            area_results[scenario] = {
                'SWE_area': area_swe,
                'SWE_polygon': polygon_swe,
                'FEGLA_area': area_model,
                'FEGLA_polygon': polygon_model
            }

        all_results[Fr_key] = area_results

    return all_results

#--------------------------------------------------------
#--------------------------------------------------------
def compute_area_errors(flood_areas_dict):
    """
    Computes relative error (%) between SWE and FEGLA areas for each Froude number.

    Parameters:
    -----------
    flood_areas_dict : dict
        Output from compute_all_flood_areas. Structure: flood_areas_dict[Fr][scenario]["SWE_area", "FEGLA_area"]

    Returns:
    --------
    error_dict : dict
        Dictionary structured as:
        error_dict[Fr] = {
            "area_error": [list of relative errors per scenario]
        }
    """
    error_dict = {}
    
    for Fr_key, scenarios in flood_areas_dict.items():
        errors = []
        for scenario_key, vals in scenarios.items():
            swe = vals["SWE_area"]
            fegla = vals["FEGLA_area"]
            
            error = abs(swe - fegla) / swe * 100
            errors.append(error)

        error_dict[Fr_key] = errors
    return error_dict


#--------------------------------------------------------
#--------------------------------------------------------
def load_flooded_ground_truth(
    city,
    Nsim=None,
    loading="calibration",       # 'calibration' | 'all'
    metadata_root = None,
    inundation_root = None,
    column="hmax"              
):
    """
    Load flooded transects and compute per-scenario area (km^2) and summaries.
    Returns: areas, polygons, transect_geometry, hmax_mean, hmax_lists, hmax_columns(or None)
    """
    # --- 1) Escenarios según modo de carga ---
    if loading == "calibration":
        pkl_path = metadata_root / f"Selected_scenarios_Nsim_{Nsim}.pkl"
        with open(pkl_path, "rb") as f:
            selected = pickle.load(f)
        scenario_ids = sorted(list(selected.keys()), key=lambda s: int(s[1:]) if s[1:].isdigit() else 10**9)
    elif loading == "all":
        base_path = inundation_root
        scenario_ids = sorted(
            [p.name for p in base_path.iterdir() if p.is_dir()],
            key=lambda s: int(s[1:]) if s[1:].isdigit() else 10**9
        )
    else:
        raise ValueError("loading must be 'calibration' or 'all'")

    # --- 2) Salidas ---
    areas = {}
    polygons = {}
    transect_geometry = {}

    # --- 3) Loop por escenario ---
    for scen_id in tqdm(scenario_ids, desc=f"Processing ({loading})"):
        scen_path = inundation_root / scen_id

        # Colección de transectos para el cómputo de área
        scen_transects = {}

        # Ordenar T*.parquet por índice numérico
        pq_files = sorted(
            scen_path.glob("T*.parquet"),
            key=lambda f: int(f.stem[1:]) if f.stem[1:].isdigit() else 10**9
        )

        for pq in pq_files:
            transect_id = pq.stem  # e.g., 'T003'
            key = f"{scen_id}_{transect_id}"

            df = pd.read_parquet(pq)

            # Geo base, guardarla una vez
            if transect_id not in transect_geometry:
                transect_geometry[transect_id] = df[["lon", "lat"]].copy()

            # Para área: lon, lat, hmax (dropna)
            scen_transects[key] = df[["lon", "lat", column]].dropna()

        # Calcular polígono y área del escenario
        if scen_transects:
            polygon_swe, area_swe = get_boundary_points(scen_transects, column=column)
            areas[scen_id] = float(area_swe) / 1e6        # m^2 -> km^2
            polygons[scen_id] = polygon_swe
        else:
            areas[scen_id] = 0.0
            polygons[scen_id] = np.nan

        # liberar referencias pesadas
        del scen_transects

    return areas, polygons, transect_geometry

#--------------------------------------------------------
#--------------------------------------------------------

def load_areas_for_mode(
    outputPath,
    Nsim,
    transect_geometry,
    mode="constant",
    name_extractor=None,
    column="height"
):
    """
    Compute areas (km^2) for a given mode: 'constant' | 'squared' | 'linear'.
    Returns dict: {name_key: {scenario_id: area_km^2}}
    Ensures transects are ordered T001, T002, ... within each scenario.
    """

    if mode not in ("constant", "squared", "linear"):
        raise ValueError("mode must be 'constant' | 'squared' | 'linear'")

    core = mode
    file_pattern = f"*{core}*{Nsim}*"

    # Name extractor per mode (consistent with original behavior)
    if mode in ("constant", "squared"):
        name_extractor = extract_cte_squared_name
    else:
        name_extractor = extract_linear_name

    results_files = list(Path(outputPath).glob(file_pattern))
    areas_by_run = {}

    # Cache lon/lat as NumPy arrays to avoid DataFrame overhead inside loops
    ll_cache = {}  # {transect_id: (lon_np, lat_np)}
    for tid, base in transect_geometry.items():
        if isinstance(base, pd.DataFrame):
            lon_np = base["lon"].to_numpy(np.float32, copy=False)
            lat_np = base["lat"].to_numpy(np.float32, copy=False)
        else:
            arr = np.asarray(base, dtype=np.float32)
            lon_np, lat_np = arr[:, 0], arr[:, 1]
        ll_cache[tid] = (lon_np, lat_np)

    _gbp = get_boundary_points  # local alias

    # Helper: extract T-number fast without regex. 'S001_T023' -> 23
    def _tnum(key: str) -> int:
        # key is 'Sxxx_Tyyy'
        return int(key.split("_", 1)[1][1:])  # take 'Tyyy' -> yyy

    for results_file in tqdm(results_files, desc=f"Loading {file_pattern} files"):
        name_key = name_extractor(str(results_file))

        with open(results_file, "rb") as f:
            data = pickle.load(f)  # {'Sxxx_Tyyy': {'height': ..., 'R0': ...}, ...}

        # Group keys by scenario once
        by_scenario = {}
        for k in data.keys():
            scen = k.split("_", 1)[0]  # 'S001'
            by_scenario.setdefault(scen, []).append(k)

        run_areas = {}

        for scenario, keys in by_scenario.items():
            # ---- build cleaned transects dict (unordered) ----
            scen_transects_unordered = {}

            for k in keys:
                t_id = k.split("_", 1)[1]          # 'T003'
                lon_np, lat_np = ll_cache[t_id]    # cached arrays

                v = data[k]
                h = v.get(column, None) if isinstance(v, dict) else None

                # normalize to float32 + pad
                if h is None:
                    hcol = np.full(lon_np.shape[0], np.nan, dtype=np.float32)
                else:
                    if isinstance(h, (list, np.ndarray)) and len(h) > 0 and isinstance(h[0], (list, np.ndarray)):
                        h = h[0]
                    h = np.asarray(h, dtype=np.float32)
                    hcol = np.full(lon_np.shape[0], np.nan, dtype=np.float32)
                    m = min(hcol.shape[0], h.shape[0])
                    if m:
                        hcol[:m] = h[:m]

                # cheap NaN filtering (avoid DataFrame.dropna)
                mask = np.isfinite(lon_np) & np.isfinite(lat_np) & np.isfinite(hcol)
                if not np.any(mask):
                    continue

                df_min = pd.DataFrame({"lon": lon_np[mask], "lat": lat_np[mask], "hmax": hcol[mask]})
                scen_transects_unordered[k] = df_min

            # ---- ensure T001..T00N order within this scenario ----
            if not scen_transects_unordered:
                run_areas[scenario] = 0.0
            else:
                # sort by transect numeric id (stable & fast)
                ordered_items = sorted(scen_transects_unordered.items(), key=lambda it: _tnum(it[0]))
                scen_transects = dict(ordered_items)  # preserves insertion order (Py 3.7+)

                # one area computation per scenario
                _, area_swe = _gbp(scen_transects, column="hmax")
                run_areas[scenario] = float(area_swe) / 1e6  # m^2 -> km^2

        areas_by_run[name_key] = run_areas
        del data, by_scenario, run_areas

    # Sort scenarios as S001, S002, ... once at the end (matches your original)
    areas_by_run = {
        run: dict(sorted(scen_areas.items(), key=lambda kv: int(kv[0][1:])))
        for run, scen_areas in areas_by_run.items()
    }

    return areas_by_run

#--------------------------------------------------------
#--------------------------------------------------------

def compute_config_area_errors_and_bias(simulation_areas, ground_truth_areas):
    """
    Compute absolute percentage error (APE, %) and signed percentage bias (Bias, %)
    for simulated areas against ground-truth areas, grouped by configuration.

    Parameters
    ----------
    simulation_areas : dict
        Structure:
        {
            'config_key_1': {'S0001': area_sim, 'S0002': area_sim, ...},
            'config_key_2': {'S0001': area_sim, 'S0002': area_sim, ...},
            ...
        }
    ground_truth_areas : dict
        Ground-truth areas by scenario:
        {
            'S0001': area_true,
            'S0002': area_true,
            ...
        }

    Returns
    -------
    error_dict : dict
        Absolute percentage error per configuration:
        {
            'config_key_1': [ape_S0001, ape_S0002, ...],
            'config_key_2': [...],
            ...
        }

    bias_dict : dict
        Signed percentage bias per configuration:
        {
            'config_key_1': [bias_S0001, bias_S0002, ...],
            'config_key_2': [...],
            ...
        }

    Notes
    -----
    - APE (%) = |A_true - A_sim| / A_true * 100
    - Bias (%) = (A_sim - A_true) / A_true * 100
    - If a scenario is missing in ground_truth_areas, it is skipped.
    - If A_true is 0 or None, NaN is appended to both lists (avoid division by zero).
    """
    error_dict = {}
    bias_dict  = {}

    for config_key, scenarios in tqdm(simulation_areas.items(), desc="Computing EA & Bias"):
        ape_list = []
        bias_list = []

        for scenario_key, sim_area in scenarios.items():
            true_area = ground_truth_areas.get(scenario_key, None)
            if true_area is None:
                # Scenario not available in ground truth -> skip
                continue

            if true_area == 0 or np.isnan(true_area):
                # Avoid division by zero / invalid true value
                ape = np.nan
                bias = np.nan
            else:
                diff = sim_area - true_area
                ape  = abs(diff) / true_area * 100.0
                bias = (diff)      / true_area * 100.0

            ape_list.append(float(ape) if np.isfinite(ape) else np.nan)
            bias_list.append(float(bias) if np.isfinite(bias) else np.nan)

        error_dict[config_key] = ape_list
        bias_dict[config_key]  = bias_list

    return error_dict, bias_dict

#--------------------------------------------------------
#--------------------------------------------------------

def load_best_transects_and_areas(city, Nsim, outputPath, transect_geometry,
                                  best_type, best_vals, column="hmax"):
    """
    Load transects and compute per-scenario flooded areas (km^2) for the winning combo.

    Returns
    -------
    name_key : str
        "F0_<F0>" for constant/squared, or "F0_<F0>_FR_<FR>" for linear.
    areas_by_scenario : dict[str, float]
        Scenario -> area in km^2.
    transects_by_scenario : dict[str, dict[str, pd.DataFrame]]
        Scenario -> { "<Sxxx>_<Tyyy>": DataFrame[lon, lat, column] }.

    Parameters
    ----------
    city : str
        City name. Also used in the results filename prefix: "{city}_...".
    Nsim : int
        Number of simulations used in the run (embedded in filename).
    outputPath : str | Path
        Base results folder; the function searches under "./Results/{city}/".
        You can pass "./Results" or "./Results/{city}" — both work.
    transect_geometry : dict[str, pd.DataFrame | array-like]
        Geometry cache mapping "Txxx" -> (lon, lat) table. Only two columns are used.
    best_type : {"constant","squared","linear"}
        Which parametrization won.
    best_vals : dict | tuple | list
        If dict: {"F0": <float>, "FR": <float optional>}
        If tuple/list: (F0, [FR,] error) — the function reads the first one or two items.
    column : str
        Name for the height column to attach (default "hmax").
    """
    if best_type not in ("constant", "squared", "linear"):
        raise ValueError("best_type must be 'constant', 'squared', or 'linear'.")

    # --- Parse F0 / FR from the input (dict or tuple/list) ---
    if isinstance(best_vals, dict):
        F0 = float(best_vals.get("F0"))
        FR = float(best_vals.get("FR")) if "FR" in best_vals else None
    else:
        F0 = float(best_vals[0])
        FR = float(best_vals[1]) if best_type == "linear" else None

    # Key used in your downstream code
    name_key = f"F0_{F0}" if best_type in ("constant", "squared") else f"F0_{F0}_FR_{FR}"

    # --- Build search path: ./Results/{city}/{city}_<type>_Nsim_<Nsim>_F0_<F0>[_FR_<FR>]_Manning_*.pkl ---
    # Accept both outputPath="./Results" and outputPath="./Results/<city>"
    city_dir = outputPath

    core = "linear" if best_type == "linear" else best_type

    # Helper: generate multiple string formats for floats to match filenames robustly
    def _fmt_variants(x: float):
        s = [
            f"{x:g}",          # compact (e.g., 1 or 0.9)
            f"{x:.1f}",        # one decimal (e.g., 1.0)
            f"{x:.2f}",        # two decimals (e.g., 0.90)
            f"{x:.3f}",        # three decimals if needed
        ]
        # Deduplicate while preserving order
        out, seen = [], set()
        for k in s:
            if k not in seen:
                out.append(k); seen.add(k)
        return out

    patterns = []
    for f0s in _fmt_variants(F0):
        if best_type == "linear":
            for frs in _fmt_variants(FR):
                patterns.append(f"{city}_{core}_Nsim_{Nsim}_F0_{f0s}_FR_{frs}_Manning_*.pkl")
        else:
            patterns.append(f"{city}_{core}_Nsim_{Nsim}_F0_{f0s}_Manning_*.pkl")

    # Search candidates (most recent wins)
    candidates = []
    for pat in patterns:
        candidates.extend(city_dir.glob(pat))
    if not candidates:
        raise FileNotFoundError(
            "No results .pkl found. Tried patterns:\n  " + "\n  ".join(str(city_dir / p) for p in patterns)
        )
    results_file = max(candidates, key=lambda p: p.stat().st_mtime)

    # --- Load the chosen results file ---
    with open(results_file, "rb") as f:
        data = pickle.load(f)  # dict: 'Sxxx_Tyyy' -> {'height': ..., 'R0': ...}

    # Group keys by scenario in a single pass
    by_scenario = {}
    for k in data.keys():
        scen_id = k.split("_", 1)[0]  # 'S001'
        by_scenario.setdefault(scen_id, []).append(k)

    FEGLA_areas = {}
    FEGLAS_polygon = {}

    # Light-weight lon/lat cache (avoid rebuilding frames repeatedly)
    lonlat_cache = {}
    for tid, base in transect_geometry.items():
        if isinstance(base, pd.DataFrame):
            lonlat_cache[tid] = base[["lon", "lat"]]
        else:
            lonlat_cache[tid] = pd.DataFrame(base, columns=["lon", "lat"])

    # Iterate scenarios ordered by numeric index (S001, S002, ...)
    for scen_id, keys in tqdm(sorted(by_scenario.items(), key=lambda kv: int(kv[0][1:])),
                              desc=f"Loading {best_type}: F0={F0}" + (f", FR={FR}" if FR is not None else "")):
        scenario_transects = {}

        # Order transects by numeric T index to keep deterministic output
        for k in sorted(keys, key=lambda s: int(s.split("_")[1][1:])):  # 'Sxxx_Tyyy'
            transect_id = k.split("_", 1)[1]                            # 'Tyyy'
            base_df = lonlat_cache[transect_id].copy(deep=False)

            v = data[k]
            h = v.get("height", None) if isinstance(v, dict) else None

            # Normalize possible shapes ([[...]] / ndarray) and pad/truncate to geometry length
            if h is None:
                hcol = np.full(len(base_df), np.nan, dtype="float32")
            else:
                if isinstance(h, (list, np.ndarray)) and len(h) > 0 and isinstance(h[0], (list, np.ndarray)):
                    h = h[0]
                h = np.asarray(h, dtype="float32")
                hcol = np.full(len(base_df), np.nan, dtype="float32")
                m = min(len(hcol), len(h))
                if m:
                    hcol[:m] = h[:m]

            df = base_df.assign(**{column: hcol})
            scenario_transects[k] = df

        # Compute area for the scenario using your boundary function
        polygon_f, area_f = get_boundary_points(
            {kk: vv[["lon", "lat", column]].dropna() for kk, vv in scenario_transects.items()},
            column=column
        )
        FEGLA_areas[scen_id] = float(area_f) / 1e6  # m^2 -> km^2
        FEGLAS_polygon[scen_id] = polygon_f
        
    return FEGLA_areas, FEGLAS_polygon

#--------------------------------------------------------
#--------------------------------------------------------

def load_hmax_to_curve(city, Nsim, metadata_path, mode="FEGLA"):
    """
    Load hmax dictionary for a city.
    mode='FEGLA' -> Selected_scenarios_Nsim_<Nsim>.pkl
    mode='all'   -> All_hmax_scenarios_Nsim_<Nsim>.pkl
    Keys are sorted numerically (S001, S002, ...).
    """
    if mode.lower() == "fegla":
        pkl = metadata_path / f"Selected_scenarios_Nsim_{Nsim}.pkl"
    elif mode.lower() == "all":
        pkl = metadata_path / f"All_hmax_scenarios_Nsim_{Nsim}.pkl"
    else:
        raise ValueError("mode must be 'FEGLA' or 'all'")

    with open(pkl, "rb") as f:
        raw = pickle.load(f)

    return dict(sorted({str(k): v for k, v in raw.items()}.items(),
                       key=lambda kv: int(kv[0][1:])))

#--------------------------------------------------------
#--------------------------------------------------------

def selecting_scenario_for_bathub(ground_truth_hmax_all, n_points=20):
    """
    Given a dict {scenario_id -> hmax}, pick n_points targets evenly spaced
    between min(hmax) and max(hmax) (inclusive) and select, for each target,
    the closest scenario (no duplicates). Returns a list of scenario_ids.
    """
    # Prepare (scenario, hmax) pairs and keep only finite values
    items = [(str(k), float(v)) for k, v in ground_truth_hmax_all.items()
             if np.isfinite(v)]
    if not items:
        raise ValueError("No valid hmax values in ground_truth_hmax_all.")

    # Sort scenarios by hmax to make selection deterministic
    items.sort(key=lambda kv: kv[1])
    scen_ids, hvals = zip(*items)
    hvals = np.asarray(hvals, dtype=float)

    # Number of picks cannot exceed number of available scenarios
    m = min(int(n_points), len(hvals))
    # Evenly spaced targets including min and max
    targets = np.linspace(hvals[0], hvals[-1], m)

    # Greedy nearest-neighbor assignment without replacement
    selected = []
    used = np.zeros(len(hvals), dtype=bool)
    for t in targets:
        # distances only for unused candidates
        d = np.where(~used, np.abs(hvals - t), np.inf)
        idx = int(np.argmin(d))
        used[idx] = True
        selected.append(scen_ids[idx])

    return selected

#--------------------------------------------------------
#--------------------------------------------------------

def compute_bathtub_area_for_list(city, scenario_ids, data_root="./Data"):
    """
    Compute bathtub area (km^2) for a list of scenarios.

    Parameters
    ----------
    city : str
        City name.
    scenario_ids : list[str]
        Scenario ids like ["S003", "S017", ...] (output from selecting_scenario_for_bathub).
    data_root : str | Path
        Base data folder containing ./Data/{city}/Flooded_transects/.

    Returns
    -------
    dict[str, float]
        {scenario_id: area_km2}
    """
    data_root = Path(data_root)
    bathtub_area = {}

    for scen_id in tqdm(scenario_ids, desc="Computing bathtub areas"):
        scen_path = data_root / city / "Flooded_transects" / scen_id

        bathtubFirstpto, bathtubLastpto = [], []

        # Iterate transects sorted by numeric T index
        pq_files = sorted(
            scen_path.glob("T*.parquet"),
            key=lambda f: int(f.stem[1:]) if f.stem[1:].isdigit() else 10**9
        )

        for pq in pq_files:
            df = pd.read_parquet(pq)

            # Distance accumulator = row index (as in your code)
            df = df.copy()
            df["cum_distance"] = df.index

            # Shoreline height (first sample)
            H0 = df["hmax"].iloc[0] if "hmax" in df.columns and len(df) else np.nan
            if pd.notna(H0):
                # User-provided function: returns X_max (horizontal reach) and (optional) Y
                X_max, _ = find_maxhorizontalflood(df, H0)

                # Find index up to X_max (robust if X_max not an exact index value)
                if X_max in df.index:
                    ix = int(df.index.get_loc(X_max))
                else:
                    # nearest index to the left
                    idx_vals = np.asarray(df.index)
                    j = int(np.searchsorted(idx_vals, X_max, side="right") - 1)
                    ix = max(1, min(j, len(df) - 1))

                dfbath = df.iloc[:ix].copy()

                # First / last point of the flooded segment
                if not dfbath.empty:
                    try:
                        bathtubFirstpto.append((float(dfbath["lon"].values[0]),
                                                float(dfbath["lat"].values[0])))
                    except Exception:
                        pass
                    try:
                        bathtubLastpto.append((float(dfbath["lon"].values[-1]),
                                               float(dfbath["lat"].values[-1])))
                    except Exception:
                        pass

            del df  # free RAM per transect

        # Polygon + area (user-provided function). If no points, area = 0.
        if bathtubFirstpto and bathtubLastpto:
            _, area_m2 = calculate_polygon_and_area(bathtubFirstpto, bathtubLastpto)
            bathtub_area[scen_id] = float(area_m2) / 1e6
        else:
            bathtub_area[scen_id] = 0.0

        # Clear per-scenario lists
        bathtubFirstpto.clear()
        bathtubLastpto.clear()

    return bathtub_area

#--------------------------------------------------------
#--------------------------------------------------------

def plot_flooding_curve(
    ground_truth_areas_all,
    ground_truth_hmax_all,
    FEGLA_areas,
    FEGLA_hmax,
    Areabath,
    Hmaxbath,
    outfigPath,
    outfile="Flooding_Curve.png"
):
    """
    Plot relationship between shoreline flood depth (Hmean) and flooded area
    for SWE simulations, FEGLA best-fit model, and bathtub approximation.
    
    Parameters
    ----------
    ground_truth_areas_all : dict
        {scenario: area_km2} for SWE simulations.
    ground_truth_hmax_all : dict
        {scenario: hmean} for SWE simulations.
    FEGLA_areas : dict
        {scenario: area_km2} for FEGLA best-fit.
    selected_scenarios : dict
        {scenario: hmean} for FEGLA best-fit scenarios.
    Hmeanbath : array-like
        Average shoreline depths for bathtub approximation.
    Areabath : array-like
        Areas corresponding to Hmeanbath.
    outfigPath : Path
        Output folder for saving the figure.
    outfile : str, optional
        Output filename. Default = "Flooding_Curve.png".
    """

    # --- SWE ---
    # Use the SAME keys to ensure alignment between area and Hmean
    sim_keys = list(ground_truth_areas_all.keys())
    AreaSim  = np.asarray([ground_truth_areas_all[k] for k in sim_keys], dtype=float)
    HmaxSim = np.asarray([ground_truth_hmax_all[k]  for k in sim_keys], dtype=float)

    # --- FEGLA (align using the same keys) ---
    fegla_keys = list(FEGLA_areas.keys())   # Define order using FEGLA_areas dict
    AreaFEGLA  = np.asarray([FEGLA_areas[k]  for k in fegla_keys], dtype=float)
    HmaxFEGLA = np.asarray([FEGLA_hmax[k] for k in fegla_keys], dtype=float)

    # Sort FEGLA by Hmean ascending (to plot a proper line)
    idx = np.argsort(HmaxFEGLA)
    HmaxFEGLA = HmaxFEGLA[idx]
    AreaFEGLA  = AreaFEGLA[idx]

    # Bathtub
    Areabath = np.array(list(Areabath.values()))
    Hmaxbath = np.array(Hmaxbath)

    # --- Figure layout ---
    fig = plt.figure(figsize=(9, 6))
    grid = plt.GridSpec(4, 4, hspace=0.15, wspace=0.15)

    # Main scatter plot
    main_ax = fig.add_subplot(grid[1:4, 0:3])
    sns.scatterplot(x=HmaxSim, y=AreaSim, ax=main_ax, s=20, c="#ffa600", label="SWE Simulations")
    sns.scatterplot(x=HmaxFEGLA, y=AreaFEGLA, ax=main_ax, s=20, c="#003f5c", label="FEGLA")
    main_ax.plot(HmaxFEGLA, AreaFEGLA, c="#003f5c", ls="--")
    main_ax.plot(Hmaxbath, Areabath, c="m", lw=1.5, ls="--", label="Bathtub")
    main_ax.set_xlabel("Average shoreline flood depth [m]", fontsize=12)
    main_ax.set_ylabel("Area [$km^2$]", fontsize=12)
    main_ax.grid(True, linestyle="--", alpha=0.7)
    main_ax.set_xticks(np.arange(0, max(HmaxFEGLA) + 1, 2))
    main_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    main_ax.legend()

    # Top histogram (Hmean distribution)
    top_ax = fig.add_subplot(grid[0, 0:3], sharex=main_ax)
    sns.histplot(HmaxSim, bins=30, kde=True, stat="density", color="gray", alpha=0.5, ax=top_ax)
    top_ax.set_ylabel("Density", fontsize=12)
    top_ax.tick_params(labelbottom=False)

    # Side histogram (Area distribution)
    side_ax = fig.add_subplot(grid[1:4, 3], sharey=main_ax)
    max_area = float(np.nanmax([np.nanmax(AreaSim), np.nanmax(AreaFEGLA)]))
    bins = np.linspace(0, max_area, 20)
    hist_FEGLA, _ = np.histogram(AreaFEGLA, bins=bins, density=True)
    hist_Sim,   _ = np.histogram(AreaSim,   bins=bins, density=True)

    # Centers and bin step
    centers = (bins[:-1] + bins[1:]) / 2
    steps   = np.diff(bins)
    if steps.size == 0:
        steps = np.array([1.0])  # fallback
    bin_step = float(np.median(steps))  # robust if hay algún bin raro

    # --- ancho automático ---
    n_series = 2                 # FEGLA y SWE
    gap_frac = 0.20              # 20% del bin para separaciones totales
    gap_abs  = gap_frac * bin_step
    bar_width = (bin_step - gap_abs) / n_series
    bar_width = max(bar_width, 1e-6)     # evitar 0
    offset = (bar_width / 2) + (gap_abs / 2)
    
    side_ax.barh(centers - offset, hist_FEGLA, height=bar_width, color="#003f5c", label="FEGLA")
    side_ax.barh(centers + offset, hist_Sim,   height=bar_width, color="#ffa600", label="SWE Simulations")
    side_ax.tick_params(labelleft=False)
    side_ax.set_xlabel("Density", fontsize=12)
    side_ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.04, 1.15))

    # Save & show
    plt.savefig(outfigPath / outfile, dpi=300, bbox_inches="tight", format="png")
    plt.show(block=False)
    plt.pause(0.1)
    
#--------------------------------------------------------
#--------------------------------------------------------
def find_min_values(
    min_F0_cte, min_err_cte,
    min_F0_sq,  min_err_sq,
    min_F0_linear, min_FR_linear, min_err_linear
):
    """
    Compara los mínimos de constant, squared y linear.
    Devuelve un diccionario con el tipo ganador y sus valores.
    """
    # Empaquetar los tres casos
    candidates = {
        "constant": {"error": min_err_cte, "F0": min_F0_cte},
        "squared":  {"error": min_err_sq,  "F0": min_F0_sq},
        "linear":   {"error": min_err_linear, "F0": min_F0_linear, "FR": min_FR_linear}
    }

    # Buscar el tipo con menor error
    best_type = min(candidates, key=lambda k: candidates[k]["error"])

    return best_type, candidates[best_type]

#--------------------------------------------------------
#--------------------------------------------------------
def _as_geom(obj):
    """
    Convert an input object into a Shapely geometry:
    - If it's already a Shapely geometry -> return it.
    - If it's a (geometry, area) tuple -> return geometry.
    - If it's a list/array of (lon, lat) coordinates -> build a Polygon.
    """
    if isinstance(obj, BaseGeometry):
        return obj
    if isinstance(obj, (tuple, list)) and len(obj) == 2 and isinstance(obj[0], BaseGeometry):
        return obj[0]
    if isinstance(obj, (list, tuple, np.ndarray)) and len(obj) >= 3:
        coords = np.asarray(obj, dtype=float)
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return Polygon(coords[:, :2])
    raise ValueError("Could not convert object to a Shapely geometry.")

def _to_multipolygon(geom):
    """Ensure that the geometry is returned as a MultiPolygon (more robust for I/O)."""
    if geom is None: 
        return None
    if isinstance(geom, MultiPolygon): 
        return geom
    if isinstance(geom, Polygon): 
        return MultiPolygon([geom])
    return geom

def _hex_to_rgba(hexstr, a=255):
    """Convert a hex color (e.g. '#ffa600') into an RGBA tuple."""
    hexstr = hexstr.lstrip('#')
    r = int(hexstr[0:2], 16)
    g = int(hexstr[2:4], 16)
    b = int(hexstr[4:6], 16)
    return r, g, b, a

def save_polygons_by_scenario(
    city: str,
    swe_polygons: dict,     # {"S001": polygon_swe, ...}
    fegla_polygons: dict,   # {"S001": polygon_fegla, ...}
    out_root: str = "./Results",
    fmt: str = "kmz",                         # 'kmz' (colored) or 'shp' (no embedded colors)
    source_crs: str | int = "EPSG:4326",
    target_crs: str | int = "EPSG:4326",
    color_map: dict = None                    # {"SWE": "#ffa600", "FEGLA": "#003f5c"}
):
    """
    Save polygons (SWE and FEGLA) into either SHP or KMZ/KML files.
    
    - If fmt='kmz', polygons are exported with embedded colors:
        SWE   -> orange (#ffa600)
        FEGLA -> blue   (#003f5c)
      This is ideal for visualization in Google Earth.
    
    - If fmt='shp', polygons are saved in a shapefile with attribute "source",
      and you can apply symbology later in QGIS/ArcGIS.

    Parameters:
    ----------
    city : str
        Name of the city (used in output folder path).
    swe_polygons : dict
        Dictionary of SWE polygons per scenario.
    fegla_polygons : dict
        Dictionary of FEGLA polygons per scenario.
    out_root : str
        Root output folder. A "Polygons" folder will be created inside ./Results/<city>.
    fmt : str
        Output format ("shp" or "kmz").
    source_crs : str or int
        CRS of input polygons (default EPSG:4326).
    target_crs : str or int
        CRS of output polygons (default EPSG:4326).
    color_map : dict
        Dictionary mapping {"SWE": hex_color, "FEGLA": hex_color}.
    """
    if color_map is None:
        color_map = {"SWE": "#ffa600", "FEGLA": "#003f5c"}

    fmt = fmt.lower()
    out_dir = Path(out_root) / "Polygons"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Union of scenario keys from both SWE and FEGLA
    scenarios = sorted(set(swe_polygons) | set(fegla_polygons),
                       key=lambda s: int(str(s)[1:]) if str(s).startswith("S") else str(s))

    if fmt == "shp":
        # Write SHP: both polygons per scenario, with "source" attribute
        for scen in tqdm(scenarios, desc="Writing SHP (no embedded colors)"):
            rows = []
            if scen in swe_polygons:
                rows.append({"scenario": scen, "source": "SWE",
                             "geometry": _to_multipolygon(_as_geom(swe_polygons[scen]))})
            if scen in fegla_polygons:
                rows.append({"scenario": scen, "source": "FEGLA",
                             "geometry": _to_multipolygon(_as_geom(fegla_polygons[scen]))})
            if not rows: 
                continue
            gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=None).set_crs(source_crs, allow_override=True)
            gdf = gdf.to_crs(target_crs)
            gdf.to_file(out_dir / f"{scen}.shp")  # symbology applied later in GIS software
    else:
        # Write KMZ with embedded styles (colors for SWE/FEGLA
        for scen in tqdm(scenarios, desc="Writing KMZ with styles"):
            kml = simplekml.Kml()
            doc = kml.newdocument(name=scen)

            def _add_mp(mp_geom, name, hexcolor):
                r, g, b, a = _hex_to_rgba(hexcolor, a=160)  # alpha=160 for semi-transparency
                kml_col = Color.rgb(r, g, b, a=a)

                fol = doc.newfolder(name=name)
                gdf = gpd.GeoDataFrame([{"geometry": mp_geom}], geometry="geometry", crs=None)
                gdf = gdf.set_crs(source_crs, allow_override=True).to_crs(target_crs)

                geom = gdf.geometry.iloc[0]
                polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
                for p in polys:
                    pol = fol.newpolygon(name=name)
                    # exterior
                    x, y = p.exterior.coords.xy
                    pol.outerboundaryis = list(zip(x, y))
                    # holes (if any)
                    for ring in p.interiors:
                        xh, yh = ring.coords.xy
                        pol.innerboundaryis.append(list(zip(xh, yh)))
                    # style
                    pol.style.polystyle.color = kml_col
                    pol.style.linestyle.color = kml_col
                    pol.style.linestyle.width = 2.0

            if scen in swe_polygons:
                _add_mp(_to_multipolygon(_as_geom(swe_polygons[scen])), "SWE", color_map["SWE"])
            if scen in fegla_polygons:
                _add_mp(_to_multipolygon(_as_geom(fegla_polygons[scen])), "FEGLA", color_map["FEGLA"])

            kml.savekmz(str(out_dir / f"{scen}.kmz"))

    print(f"Saved to: {out_dir.resolve()}")