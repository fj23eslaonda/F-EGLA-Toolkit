# ==============================================================================
# FEGLA runner
#
# File Name    : FEGLA_runner.py
# Author       : Francisco J. Sáez
# Affiliation  : Research Center for Integrated Disaster Risk Management (CIGIDEN)
# GitHub       : https://github.com/fj23eslaonda
#
# Description  : Python script for applying the FEGLA (Forward Energy Grade Line 
#                Analysis) method to estimate tsunami-induced flooding across 
#                transects. Supports full dataset application and calibration 
#                workflows using variable Froude number parameterizations 
#                (linear, squared, constant).
#
# Created On   : 2025-02-01
# Last Updated : 2025-06-09
# Version      : 2.0.0
#
# Usage        : Designed for scientific research and academic purposes.
#                Requires configuration via JSON input file with model settings.
#
# License      : MIT License
# ==============================================================================

#--------------------------------------------------------
# Packages
#--------------------------------------------------------
import multiprocessing
import numpy as np
import gc
import pickle
from tqdm import tqdm
from pathlib import Path

from tsunamicore.fegla.operations import (
    transect_processing, 
    find_maxhorizontalflood,
    create_batches_from_combinations,
    load_selected_scenarios_from_pickle,
    load_selected_scenarios_from_parquet
    )
    
#--------------------------------------------------------
# Model constant
#--------------------------------------------------------
g              = 9.81  # gravity constant
delta_x        = -1.0  # delta x for computations
tolerance      = 0.01  # error tolerance for convergence
max_iterations = 100   # maximum number of iterations allowed

#--------------------------------------------------------
# Wrapper function to parallelize transect processing
#--------------------------------------------------------
def process_transect_wrapper(args):
    idx, dataframes, F0, FR, mode, hmean_value = args
    return process_transect(idx, dataframes, F0, FR, mode, hmean_value)

#--------------------------------------------------------
# Main function to process a single transect
#--------------------------------------------------------
def process_transect(idx, scen_tran_all, F0, FR, mode, hmean_value, message=True):
    if message:
        print('================')
        print('Processing :', idx)
    
    # Initialize iteration lists for each transect
    height_iteration = []
    all_XRmax        = []
    all_XRmin        = []
    all_R0           = []
    error_iteration  = []

    flood_transect = scen_tran_all[idx]
    iteration      = 0

    # Extract initial variables
    z_initial          = flood_transect['elevation'].values
    manning_initial    = flood_transect['manning'].values
    h0                 = float(hmean_value)
    distance_initial   = flood_transect['cum_distance'].values

    # Check if all hmax values are NaN (Transect Not Flooded)
    if np.all(np.isnan(h0)):
        if message:
            print(f"Transect {idx} was NOT flooded. Returning NaN values.")
        return idx, {
            'height': np.full((1, len(z_initial)), np.nan, dtype=np.float32),
            'error': np.array([np.nan], dtype=np.float32),
            'XRmax': np.array([np.nan], dtype=np.float32),
            'XRmin': np.array([np.nan], dtype=np.float32),
            'R0': np.nan
        }

    # Continue processing if the transect was flooded
    R0   = h0 + 0.5 * F0**2 * h0

    # Find initial X_max and X_min
    X_max, _ = find_maxhorizontalflood(flood_transect, R0)
    X_min    = 0
    X_R      = np.mean([X_max, X_min])

    stagnation_counter = 0
    previous_error = None

    while iteration < max_iterations:

        X_R_used = X_R  # Store XR as fixed for this iteration (avoid mid-iteration changes)

        # Calculate number of segments (N) and dx spacing based on XR
        N = int(np.round(X_R_used))
        delta_x = - X_R_used / N  # Negative because the integration moves leftward

        # Create new distance array with the updated dx
        new_distance = np.linspace(0, X_R_used, N + 1)

        # Interpolate elevation and Manning values onto the new distance grid
        z = np.interp(new_distance, distance_initial, z_initial)
        manning = np.interp(new_distance, distance_initial, manning_initial)

        # Trim distance to match XR
        distance_dummy = new_distance[new_distance <= X_R_used]

        # Update local Froude numbers across the profile
        if mode == 'linear':
            Fr = F0 + (FR - F0) * distance_dummy / X_R_used
        elif mode == 'squared':
            Fr = F0 * (distance_dummy / X_R_used) ** 2
        elif mode == 'constant':
            Fr = np.full_like(distance_dummy, F0)
        else:
            raise ValueError(f"Invalid froude_mode: {mode}")

        # Initialize water depth and velocity arrays
        h_opti_list = np.full(distance_dummy.size, np.nan)
        velocity_list = np.full(distance_dummy.size, np.nan)
        h_opti_list[-1] = 0.01  # Boundary condition at the run-up end

        # Integrate from right (runup) to left (towards shoreline)
        for i in range(len(distance_dummy) - 1, 0, -1):
            z_next, z_prev = z[i - 1], z[i]
            h_prev = h_opti_list[i]
            Fr_next, Fr_prev = Fr[i - 1], Fr[i]
            manning_coeff = manning[i - 1]

            # Velocity based on Froude number and depth
            u_prev = Fr_prev * np.sqrt(g * h_prev) if h_prev >= 0 else 0
            velocity_list[i] = u_prev

            # Compute energy and losses
            energy_i = z_prev + h_prev + 0.5 * Fr_prev**2 * h_prev
            loss_i = (g * Fr_next**2 * manning_coeff**2 * delta_x) / (h_prev**(1/3)) if h_prev > 0 else 0

            # Estimate next water depth
            h_next = 1 / (1 + 0.5 * Fr_next**2) * (energy_i - loss_i - z_next)
            h_next = max(h_next, 0)  # Ensure non-negative depth

            h_opti_list[i - 1] = h_next
            velocity_list[i - 1] = Fr_next * np.sqrt(g * h_next) if h_next >= 0 else 0

        # Compute relative error in water depth at starting point
        h0_opti = h_opti_list[0]
        h0_error = np.abs(h0_opti - h0) / h0
        error_iteration.append(h0_error)

        # Store iteration values
        height_iteration.append(np.array(h_opti_list.copy(), dtype=np.float32))
        all_XRmax.append(X_max)
        all_XRmin.append(X_min)
        all_R0.append(R0)

        # Check convergence based on tolerance
        if np.round(h0_error, 3) <= tolerance:
            break

        # Update XR bounds for the next iteration
        if h0_opti < h0:
            X_min = X_R_used
        else:
            X_max = X_R_used

        # Adjust R0 only if the error difference is small for 5 consecutive iterations
        if previous_error is not None:
            error_diff = np.abs(previous_error - h0_error)
            if error_diff < 0.001:
                stagnation_counter += 1
            else:
                stagnation_counter = 0  # reset if error difference is larger again

            if stagnation_counter >= 5:
                if message:
                    print("Adjusting R0 by 10% after 5 stagnant error iterations.")
                if h0_opti >= h0:
                    R0 *= 0.9
                else:
                    R0 *= 1.1
                X_max, _ = find_maxhorizontalflood(flood_transect, R0)
                X_min = 0  # Reset lower bound
                stagnation_counter = 0  # reset counter after applying change

        # Prepare XR for next iteration
        X_R = 0.5 * (X_max + X_min)
        previous_error = h0_error
        iteration += 1
    
    if message:
        if np.round(h0_error,3) > 0.01:
            print(f'height error : {h0_error:.3f}, iter : {iteration}')
        else:
            # Print iteration diagnostics
            print(f'h0_error: {h0_error:.3f}, h0: {h0:.3f}, h0_opti:{h0_opti:.3f}, iter : {iteration}')

    # Return all the results
    return idx, {
        'height': height_iteration[-1].astype(np.float32).tolist(),  # only if you need to serialize
        'error': float(np.float32(error_iteration[-1])),
        'XRmax': float(np.float32(all_XRmax[-1])),
        'XRmin': float(np.float32(all_XRmin[-1])),
        'R0': float(np.float32(all_R0[-1]))
    }

# --------------------------------------------------------
# Calibration model
# Main function to process all transects in parallel
# --------------------------------------------------------
def FEGLA_calibration(params, scenarios_to_run=None, transect_to_run=None):

    # Extract required parameters from the JSON input
    city        = params['city']
    batch_size  = params['batch_size']
    manning     = params['manning']
    selected_scenarios =  params['selected_scenarios']

    # get project root directory
    root = Path(__file__).resolve().parents[2]
    # get metadata path
    meta_path = root / 'outputs' / city / 'metadata'
    # get flooded transects path
    inundation_path = root / 'outputs' / city / 'inundation'
    # Save model output for current combination
    results_folder = root / 'outputs' / city / 'calibration'
    results_folder.mkdir(parents=True, exist_ok=True)

    # Load list of F0 values (required)
    F0_list = params.get('F0')
    if not F0_list:
        raise ValueError("You must provide at least one F0 value.")

    # Load FR list (required only for linear mode)
    FR_list = params.get('FR', [])

    # ---------------------------------
    # Test all Froude parameterizations
    # ---------------------------------
    modes_to_test = ['linear', 'squared', 'constant']
    param_combinations = []

    for mode in modes_to_test:
        if mode == 'linear':
            if not FR_list:
                raise ValueError("FR list is required for calibration in 'linear' mode.")
            # All combinations of F0 and FR
            combinations = [(mode, F0, FR) for F0 in F0_list for FR in FR_list]
        else:
            # For squared/constant, FR is not used
            combinations = [(mode, F0, None) for F0 in F0_list]
        param_combinations.extend(combinations)
    
    # ------------------------------------------
    # Load scenarios (selected or entire folder)
    # ------------------------------------------
    if scenarios_to_run is None:
        # Load representative scenarios for calibration
        pkl_file = meta_path / selected_scenarios
        scenarios_to_run, hmean = load_selected_scenarios_from_pickle(pkl_file)
        scenario_to_hmean = dict(zip(scenarios_to_run, hmean))
        if not scenarios_to_run:
            raise ValueError("No selected scenarios found for calibration.")

    Nsim = len(scenarios_to_run)

    # Load all corresponding transect DataFrames
    all_data = load_selected_scenarios_from_parquet(inundation_path, scenarios_to_run)
    all_keys = list(all_data.keys())

    # Optional: filter to specific transects
    if transect_to_run is not None:
        all_keys = [k for k in all_keys if k.split('_')[1] in transect_to_run]

    if not all_keys:
        raise ValueError("No valid combinations of scenarios and transects found.")

    # Divide transects into batches for multiprocessing
    batch_list = create_batches_from_combinations(all_keys, batch_size)

    # ------------------------------------------
    # Run FEGLA model for each parameter setting
    # ------------------------------------------
    for mode, F0, FR in param_combinations:
        print(f"\nRunning model for mode = {mode}, F0 = {F0}, FR = {FR if FR is not None else 'N/A'}")
        all_results = {}

        for batch in tqdm(batch_list, desc="Processing batches", total=len(batch_list)):
            # Select current batch of transects
            dataframes = {k: all_data[k] for k in batch}
            idx_batch = list(dataframes.keys())

            # Preprocess transects (e.g., apply Manning coefficient)
            dataframes = transect_processing(dataframes, idx_batch, manning_coeff=manning)  

            # Parallel execution of FEGLA for each transect
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                args = []
                for idx in idx_batch:
                    scenario = idx.split('_')[0]  # e.g., "S0040"
                    hmean_value = scenario_to_hmean.get(scenario, None)
                    if hmean_value is None:
                        raise ValueError(f"No hmean found for scenario {scenario}")
                    args.append((idx, dataframes, F0, FR if FR is not None else F0, mode, hmean_value))
                results = pool.map(process_transect_wrapper, args, chunksize=12)

            # Store results
            for idx, res in results:
                if res is not None:
                    all_results[idx] = res

            del dataframes
            gc.collect()

        # Build the filename
        output_name = f'{city}_{mode}_Nsim_{Nsim}_F0_{F0}' + (f'_FR_{FR}' if FR is not None else '') + f'_Manning_{manning}.pkl'
        output_file = results_folder / output_name

        with open(output_file, 'wb') as f:
            pickle.dump(all_results, f)

        print(f"Saved results to {output_file}")
