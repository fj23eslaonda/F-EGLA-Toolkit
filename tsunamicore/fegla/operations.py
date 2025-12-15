#--------------------------------------------------------
#
# PACKAGES
#
#--------------------------------------------------------
import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path
from tqdm import tqdm

#--------------------------------------------------------
#
# FUNCTIONS
#
#--------------------------------------------------------
def filter_elevation(df, max_threshold=30):
    """
    Filters the transect DataFrame to retain rows up to the first instance where the elevation exceeds max_threshold.

    Parameters:
    df (DataFrame): The DataFrame containing transect data with an 'Elevation' column.
    max_threshold (float): The maximum elevation threshold.

    Returns:
    DataFrame: The filtered DataFrame with rows where the elevation is less than or equal to max_threshold 
               and includes rows up to the first instance where the elevation exceeds max_threshold.
    """
    # Find the first index where the elevation exceeds max_threshold
    above_threshold_idx = df.index[df['elevation'] > max_threshold]
    
    if not above_threshold_idx.empty:
        # Return all rows up to and including the first row that exceeds max_threshold
        return df.iloc[:above_threshold_idx[0] + 1].reset_index(drop=True)
    
    # If no elevation exceeds max_threshold, return the entire DataFrame
    return df.reset_index(drop=True)

#-------------------------------------------------------
#-------------------------------------------------------

def compute_cumulative_distance(df):
    """
    Computes an approximate cumulative distance along the transect using Euclidean distance approximation.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'lat' and 'lon' columns.

    Returns:
    - pd.DataFrame: Updated DataFrame with 'cumulative_distance' column.
    """
    LAT_TO_METERS = 111_000  # 1 degree of latitude ~ 111 km
    LON_TO_METERS = 111_000  # Adjust based on latitude (simplified for now)

    lat_diff = np.diff(df["lat"].values) * LAT_TO_METERS  # Convert lat degrees to meters
    lon_diff = np.diff(df["lon"].values) * LON_TO_METERS  # Convert lon degrees to meters

    distances = np.sqrt(lat_diff**2 + lon_diff**2)  # Compute Euclidean distance
    cumulative_distance = np.insert(np.cumsum(distances), 0, 0)  # Add zero for first point
    
    return cumulative_distance

#-------------------------------------------------------
#-------------------------------------------------------

def fill_initial_nans(df, column):
    """
    Fills the initial NaN values in the specified column of the DataFrame with the first non-NaN value.
    Leaves trailing NaN values unchanged.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    column (str): The name of the column to fill NaN values in.

    Returns:
    DataFrame: The DataFrame with the initial NaN values in the specified column filled.
    """
    # Find the first non-NaN value index
    first_valid_index = df[column].first_valid_index()

    if first_valid_index is not None:
        # Fill NaNs before the first valid value using loc to avoid chained assignment
        df.loc[:first_valid_index, column] = df[column].iloc[first_valid_index]

    return df

#-------------------------------------------------------
#-------------------------------------------------------

def transect_processing(scen_tran_all, idx_batch, manning_coeff, flood=True):
    """
    Processes transects from the dataset by applying a height threshold filter, calculating cumulative distance,
    and assigning a uniform Manning's coefficient.

    Parameters:
    scen_tran_all (dict): The dataset containing transect information, structured as a dictionary of dictionaries.
    scen_idx (list): The list of scenario indices to process.
    tran_idx (list): The list of transect indices to process.
    manning_coeff (float): The Manning's coefficient to be applied uniformly across the transect.
    threshold (float): The elevation threshold to filter the transect data.

    Returns:
    dict: A dictionary containing the processed transect data including coordinates,
          elevation, cumulative distance, and Manning's coefficient.
    """
    processing_dict = {}
    # Iterate over scenarios and transects
    for idx in idx_batch:
        # Work on a copy of the current transect DataFrame
        df = scen_tran_all[idx].copy()

        # Calculate flood levels, cumulative distance, and assign Manning's coefficient
        if flood:
            df['flood'] = df['elevation'] + df['hmax']
        df['cum_distance'] = compute_cumulative_distance(df)
        df['manning'] = manning_coeff

        # Save the processed DataFrame back to the main dictionary
        processing_dict[idx] = df

    return processing_dict

#-------------------------------------------------------
#-------------------------------------------------------

def get_XR_and_R(scen_tran):
    '''
    Extracts the XR and R values from a given transect in a specified scenario.

    Parameters:
    - scen_tran_all: dict, a nested dictionary where the keys are scenario identifiers (e.g., 'S001') and the values are dictionaries
      containing transects, with each transect represented as a DataFrame that includes columns like 'cum_distance' and 'height'.
    - idx_batch: str, the identifier for the specific transect within the scenario to be processed.

    Returns:
    - tuple, containing:
      - XR: float, the cumulative distance ('cum_distance') at the point where 'height' first exceeds zero when traversing the DataFrame from the end to the beginning.
      - R: float, the elevation ('elevation') corresponding to the same point where 'height' first exceeds zero.
    '''
    # Access the DataFrame for the current scenario and transect
    df_dummy = scen_tran
    
    # Reverse the DataFrame and find the first index where 'height' is greater than 0
    reversed_df = df_dummy.iloc[::-1]
    positive_height_index = reversed_df[reversed_df['height'] > 0].index[0]
    
    # Extract the corresponding values of 'XR' and 'R'
    XR = df_dummy.loc[positive_height_index, 'cum_distance']
    R  = df_dummy.loc[positive_height_index, 'elevation']

    return R, XR


#-------------------------------------------------------
#-------------------------------------------------------

def energy_balance(h_next, h_prev, z_next, z_prev, Fr_next, Fr_prev, g, n, delta_x):
    """
    Computes the difference in energy between two consecutive points in a transect for the purpose of 
    hydrodynamic modeling. The energy balance is based on the Bernoulli equation, adapted for a channel flow 
    scenario with considerations for gravitational force, Froude number, and Manning's roughness coefficient.

    Parameters:
    - h_next: float, water depth at the next point (i+1).
    - h_prev: float, water depth at the previous point (i).
    - z_next: float, bed elevation at the next point (i+1).
    - z_prev: float, bed elevation at the previous point (i).
    - Fr_next: float, Froude number at the next point (i+1).
    - Fr_prev: float, Froude number at the previous point (i).
    - g: float, acceleration due to gravity.
    - n: float, Manning's roughness coefficient.
    - delta_x: float, distance between the two points (i and i+1).

    Returns:
    - float: The difference in energy between the next point (i+1) and the previous point (i). 
             This value is used to assess whether the energy at these points balances according to the 
             assumptions of the flow model.
    """
    
    # Calculate the energy at the previous point (i)
    E_prev = z_prev + h_prev + 0.5 * Fr_prev**2 * h_prev
    
    # Calculate the energy at the next point (i+1)
    # The term (g * Fr_next**2 * n**2 * delta_x) / (h_next**(1/3)) accounts for energy losses due to friction
    E_next = z_next + h_next + 0.5 * Fr_next**2 * h_next + (g * Fr_next**2 * n**2 * delta_x) / (h_next**(1/3))
    
    # Return the difference in energy between the next point (E_next) and the previous point (E_prev)
    return E_next - E_prev

#-------------------------------------------------------
#-------------------------------------------------------

def find_maxhorizontalflood(df, R):
    """
    Find the first crossing point where the elevation curve intersects R (runup level).
    If no crossing is found, return the last point of the transect.

    Parameters:
    df (DataFrame): Must contain 'elevation' and 'cum_distance'.
    R (float): Runup level.

    Returns:
    tuple: (cum_distance, index) of the crossing or last point if no crossing is found.
    """
    diff = df['elevation'].values - R

    for i in range(1, len(diff)):
        if diff[i-1] * diff[i] <= 0:  # Crossing or touching R0
            return df['cum_distance'].iloc[i], df.index[i]

    # If no crossing is found, return the last point
    last_idx = df.index[-1]
    return df['cum_distance'].iloc[-1], last_idx

#-------------------------------------------------------
#-------------------------------------------------------
def find_steeper_slope(pos, elev):
    """
    Determines the height of the wall based on the slope of the terrain.
    
    Args:
        pos (list or array): List of positions (cumulative distance).
        elev (list or array): List of corresponding elevations.
    
    Returns:
        float: Wall height or 0 if no valid wall is found.
    """
    wall_height = 0
    wall_position = 0
    n_gentle_slopes = 0
    first_steep_slope = False
    length = len(pos)
    
    for i in range(1, length):
        # Verificar si hay un paso cero entre posiciones consecutivas
        delta_pos = pos[i] - pos[i-1]
        
        if delta_pos == 0:
            continue  # Si hay una división por cero, se omite esta iteración
        
        slope = (elev[i] - elev[i-1]) / delta_pos
        
        if slope > 0.06:
            # If this is the first steep slope
            if not first_steep_slope:
                first_steep_slope = True
            # Update wall height and position
            wall_height = elev[i]
            wall_position = pos[i]
            n_gentle_slopes = 0
        else:
            # If there is a gentle slope
            n_gentle_slopes += 1
            if first_steep_slope and n_gentle_slopes >= 15 and wall_position < 300:
                return wall_height

    # If the wall position exceeds 300 or the condition is not met
    return wall_height if wall_position < 300 else 0

#-------------------------------------------------------
#-------------------------------------------------------
def create_batches_from_combinations(combinations, batch_size):
    """
    Creates a list of batches where each batch contains up to `batch_size` combinations.
    Each combination corresponds to a specific scenario and transect pair.

    Parameters:
    - combinations (List[str]): The list of valid scenario-transect combinations.
    - batch_size (int): The number of combinations in each batch.

    Returns:
    - List[List[str]]: A list of batches, where each batch is a list of combinations.
    """
    # Create batches by slicing the combinations list into chunks of size `batch_size`
    batches = [combinations[i:i + batch_size] for i in range(0, len(combinations), batch_size)]
    
    return batches

#-------------------------------------------------------
#-------------------------------------------------------
def load_batch_from_hdf5(filepath, batch_keys):
    """
    Loads a batch of DataFrames from an HDF5 file based on a list of keys.

    Parameters:
    - filepath (str): The path to the HDF5 file.
    - batch_keys (List[str]): A list of keys (scenario_transect combinations) to load from the HDF5 file.

    Returns:
    - dict: A dictionary where keys are the scenario_transect identifiers and values are the corresponding DataFrames.
    """
    batch_data = {}

    # Open the HDF5 file
    with pd.HDFStore(filepath, mode='r') as store:
        # Load each DataFrame corresponding to the keys in batch_keys
        for key in batch_keys:
            if key in store:
                batch_data[key] = store[key]
            else:
                print(f"Warning: Key '{key}' not found in the HDF5 file.")

    return batch_data

#-------------------------------------------------------
#-------------------------------------------------------
def extract_nsim_value(file_path):
    """
    Extracts the numeric value after 'Nsim_' in the file name.
    
    Args:
        file_path (str or Path): Path to the file.

    Returns:
        int: The extracted number (e.g. 100 from 'Nsim_100'), or None if not found.
    """
    match = re.search(r'Nsim_(\d+)', file_path.name)
    return int(match.group(1)) if match else None

#-------------------------------------------------------
#-------------------------------------------------------
# Helper: Load selected scenarios and their transects
def load_selected_scenarios_from_parquet(base_path, selected_scenarios):
    """
    Load transect DataFrames from parquet files for given scenario IDs.

    Parameters:
    - base_path (Path): Path to base ./Data/{city}/Flooded_transects/
    - selected_scenarios (list): List of scenario IDs (e.g., ['S0025', 'S0140'])

    Returns:
    - dict: Dictionary { 'Sxxxx_Tyyy': DataFrame }
    """
    data = {}
    base_path = Path(base_path)

    for scenario_id in tqdm(selected_scenarios, desc="Loading selected scenarios"):
        scenario_folder = base_path / scenario_id
        if not scenario_folder.exists():
            print(f"Warning: Scenario folder not found: {scenario_folder}")
            continue

        for file in scenario_folder.glob("*.parquet"):
            transect_id = file.stem  # e.g., T001
            key = f"{scenario_id}_{transect_id}"
            df = pd.read_parquet(file)
            data[key] = df

    return data

#-------------------------------------------------------
#-------------------------------------------------------
# Helper to load selected scenarios from pkl
def load_selected_scenarios_from_pickle(pkl_path):
    """
    Load a list of scenario IDs from a pickle file.

    Parameters:
    - pkl_path (str): Path to the .pkl file.

    Returns:
    - list of scenario IDs if successful, otherwise None.
    """
    try:
        with open(pkl_path, "rb") as f:
            allscenarios = pickle.load(f)
            scenarios = [str(key) for key in allscenarios.keys()]
            hmean     = [value for value in allscenarios.values()]
            if not isinstance(scenarios, list):
                raise ValueError("Pickle content is not a list.")
            return scenarios, hmean
    except Exception as e:
        print(f"[Warning] Could not load scenarios from '{pkl_path}': {e}")
        return None