
#--------------------------------------------------------
# PACKAGES
#--------------------------------------------------------
import xarray as xr
import argparse
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

from tsunamicore.preprocessing.simulation import (
    concatenate_transects,
    interpolate_simulations_to_xarray,
    compute_mean_hmax_at_shoreline,
    select_representative_scenarios_with_extremes,
    plot_scenario_selection, 
    extract_selected_scenarios_to_h5,
    extract_all_scenarios_to_parquet,
    plot_random_transects,
)

#--------------------------------------------------------
# Helpers
#--------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

#--------------------------------------------------------
# Main function
#--------------------------------------------------------
def main(city, n_selected_sim):
    """
    Processes tsunami simulation outputs to compute maximum inundation along
    predefined shoreline transects.

    This function loads Hmax simulation files, interpolates their values over all
    transect points, computes summary metrics (e.g., mean Hmax at shoreline), and
    optionally selects representative scenarios for calibration. It also saves the
    processed datasets and produces diagnostic plots to support the FEGLA workflow.

    Parameters
    ----------
    city : str
        Name of the study site.
    calibration : bool
        If True, selects and exports representative scenarios; if False, only
        computes summary metrics and plots.
    n_selected_sim : int
        Number of representative simulations to extract when calibration is enabled.
    """
    root = Path(__file__).resolve().parent.parent
    data_path = root / 'data' / city
    output_path = root / 'outputs' / city 
    outfig_path = output_path / 'figs'
    transect_path = output_path / 'transects'

    # Creating metadata folder
    meta_path = output_path / 'metadata'
    meta_path.mkdir(parents=True, exist_ok=True)

    # Creating inundation folder to storage flooded transects
    inundation_path = output_path / 'inundation'
    inundation_path.mkdir(parents=True, exist_ok=True)

    # Load flood maps
    print('\nLoading data from simulations...')
    hmax_files = sorted(data_path.glob("hmax*.nc"))
    flood_maps = {filepath.name: xr.open_dataset(filepath) for filepath in hmax_files}

    # Load transect data
    print('\nLoading transects...')
    with pd.HDFStore(transect_path / 'Transects_data.h5', mode="r") as h5_store:
        transectData = {key.lstrip('/'): h5_store[key] for key in h5_store.keys()}
    print(f'N° of transects: {len(transectData.keys())}')

    transectData = {key: df.reset_index(drop=True) for key, df in transectData.items()}
    all_transect_points = concatenate_transects(transectData)

    # Interpolate simulations and create the xarray.Dataset
    print('\nInterpolating transects over simulation...')
    ds = interpolate_simulations_to_xarray(flood_maps, all_transect_points)

    # Compute mean hmax at shoreline
    print('\nComputing mean hmax for all scenarios...')
    scenario_mean_hmax = compute_mean_hmax_at_shoreline(ds)

    with open(meta_path / f"All_hmax_scenarios_Nsim_{n_selected_sim}.pkl", "wb") as f:
            pickle.dump(scenario_mean_hmax, f)
    
    # Select representative scenarios
    selected_scenarios = select_representative_scenarios_with_extremes(scenario_mean_hmax, n_selected_sim, n_extreme=5, extreme_mode='top')
    with open(meta_path / f"Selected_scenarios_Nsim_{n_selected_sim}.pkl", "wb") as f:
        pickle.dump(selected_scenarios, f)

    # Plot scenario selection
    plot_scenario_selection(scenario_mean_hmax, selected_scenarios, outfig_path, n_selected_sim)

    # Extract selected scenarios to HDF5 file
    results = extract_selected_scenarios_to_h5(ds, selected_scenarios)

    # Plot randomly selected scenarios
    plot_random_transects(results, outfig_path, n_selected_sim)

        
    print('\nSaving all flooded transects as h5 file ...')
    extract_all_scenarios_to_parquet(ds, inundation_path, n_jobs=-1)

    # Ask user whether to exit
    input("\nPress Enter to finish...")
    plt.close('all')
    print("\nProcess completed.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process flood simulation data.")
    parser.add_argument("--city", type=str, required=True, help="City name for data processing.")
    parser.add_argument("--n_selected_sim", type=int, required=True,
                    help="Number of selected simulations (required if Calibration=True).")

    args = parser.parse_args()

    main(args.city, args.n_selected_sim)
