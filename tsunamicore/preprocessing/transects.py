#--------------------------------------------------------
#
# PACKAGES
#
#--------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from skimage import measure
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib import ticker
import matplotlib.patheffects as pe

from tsunamicore.utils.plot_style import apply_plot_style
apply_plot_style()

#--------------------------------------------------------
#
# FUNCTIONS
#
#--------------------------------------------------------

def plot_topobathymetry_and_contours(
    x, y, z,
    z0_contour=None,
    alpha=0.7,
    ax=None
):
    """
    Plot a topobathymetry map with predefined CUSTOM_LEVELS,
    using discrete colormap bins for bathymetry and topography.
    A contour for z=0 can also be plotted if provided.

    Parameters:
    x (np.array): 2D array of longitude values for each grid point.
    y (np.array): 2D array of latitude values for each grid point.
    z (np.array): 2D array of elevation data at the grid points defined by x and y.
    z0_contour (np.array): Optional. 2D array of x, y coordinates for the z=0 contour line.
    alpha (float): Transparency for filled contours.
    ax (matplotlib.axes._subplots.AxesSubplot): Optional axis to plot on.

    Returns:
    fig (matplotlib.figure.Figure): Matplotlib Figure object.
    ax (matplotlib.axes._subplots.AxesSubplot): Axis with plot.
    cp (QuadContourSet): Contourf object for further customization.
    """

    # --- Custom discrete levels (can be tuned as needed) ---
    CUSTOM_LEVELS = [-100, -50, -30, -20, -10, 0, 5, 10, 20, 30, 50, 100]

    # Split levels into bathy (<0) and topo (>0)
    i0 = CUSTOM_LEVELS.index(0)
    k_neg = i0
    k_pos = len(CUSTOM_LEVELS) - 1 - i0

    # Define colormaps for bathy and topo
    bathy_cmap = plt.cm.GnBu   # green-blue for bathymetry
    topo_cmap  = plt.cm.gist_gray_r  # gray for topography

    blues  = bathy_cmap(np.linspace(0, 1, k_neg)) if k_neg > 0 else np.empty((0, 4))
    greens = topo_cmap (np.linspace(0, 1, k_pos)) if k_pos > 0 else np.empty((0, 4))
    colors = np.vstack([blues, greens])

    listed = mcolors.ListedColormap(colors, name="bathy_topo_steps")
    listed.set_under(colors[0])    # everything < min level stays first color
    listed.set_over(colors[-1])    # everything > max level stays last color

    norm = mcolors.BoundaryNorm(CUSTOM_LEVELS, ncolors=listed.N, clip=True)

    # --- Initialize figure/axes ---
    create_new_fig = ax is None
    if create_new_fig:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    # --- Filled contours with discrete cmap ---
    cp = ax.contourf(
        x, y, z,
        levels=CUSTOM_LEVELS,
        cmap=listed,
        norm=norm,
        alpha=alpha,
        extend='both'
    )

    # --- Contour lines ---
    cs = ax.contour(x, y, z, levels=CUSTOM_LEVELS, colors='black', linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f m')

    # # --- Colorbar aligned with axis ---
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%", pad=0.05)
    # cbar = fig.colorbar(cp, cax=cax, extend='both', extendrect=True)
    # cbar.set_label("Elevation [m]")
    # cbar.set_ticks(CUSTOM_LEVELS)

    # --- Shoreline at z=0 (if provided) ---
    if z0_contour is not None:
        ax.plot(z0_contour[:, 0], z0_contour[:, 1], c='k', label='Shoreline [0 m]')
        ax.legend()

    # --- Titles and labels ---
    ax.set_title('Topobathymetry Map')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')

    fig.tight_layout()

    if create_new_fig:
        return fig, ax, cp
    return ax, cp

#-------------------------------------------------------
#-------------------------------------------------------
def find_contour_coordinates(x, y, z, level=0):
    """
    Finds the geographic coordinates of a contour at a specified elevation level 
    within a dataset where x, y represent longitude and latitude respectively, 
    and z represents elevation.
    
    Parameters:
        x (np.array): 2D array of longitude values for each grid point.
        y (np.array): 2D array of latitude values for each grid point.
        z (np.array): 2D array representing elevation data at the grid points defined by x and y.
        level (float): The elevation level at which to find the contour.
    
    Returns:
        np.array: A 2D array where each row contains the geographic coordinates 
        [longitude, latitude] of points along the contour of the specified elevation level.
    """
    
    # Find the contours at the specified elevation level
    contours = measure.find_contours(z, level)
    
    # Initialize an empty list to store the transformed coordinates
    coordinates = []

    # Iterate over each contour
    for contour in contours:
        for point in contour:
            # Los contornos devuelven los índices, no las coordenadas en lon/lat
            i, j = int(point[0]), int(point[1])
            lon = x[j]
            lat = y[i]
            coordinates.append((lon, lat))

    # Convertir a numpy array para su manipulación
    coordinates = np.array(coordinates)
    
    return coordinates

#-------------------------------------------------------
#-------------------------------------------------------
def moving_average_extended(data, window_size):
    """ Apply a simple moving average filter to the data, preserving the original length. """
    padded_data = np.pad(data, (window_size//2, 
                                window_size-1-window_size//2), mode='edge')
    smoothed_data = np.convolve(padded_data, 
                                np.ones(window_size)/window_size, mode='valid')
    return smoothed_data

#-------------------------------------------------------
#-------------------------------------------------------
def create_spline_function(shoreline, n_point=1000, smooth_window=20):
    """
    Creates a cubic spline interpolation function for a given shoreline curve using arc length parameterization.
    """
    # Separate the x and y coordinates
    x = shoreline[:, 0]
    y = shoreline[:, 1]

    # Calculate the arc length of each segment
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    arc_lengths = np.concatenate(([0], np.cumsum(distances)))

    # Filter out duplicate arc lengths
    unique_arc_lengths, unique_indices = np.unique(arc_lengths, return_index=True)
    x_unique = x[unique_indices]
    y_unique = y[unique_indices]
    print(f"Unique arc lengths: {len(unique_arc_lengths)}")

    if len(unique_arc_lengths) < 2:
        raise ValueError("Not enough unique points for spline interpolation. Check input data or reduction factor.")

    # Create the cubic spline interpolation functions
    spline_function_x = CubicSpline(unique_arc_lengths, x_unique)
    spline_function_y = CubicSpline(unique_arc_lengths, y_unique)

    # Generate a dense set of arc length points for a smooth curve
    arc_lengths_dense = np.linspace(unique_arc_lengths[0], unique_arc_lengths[-1], num=n_point)

    # Evaluate the spline functions at the dense arc length points
    x_smooth = spline_function_x(arc_lengths_dense)
    y_smooth = spline_function_y(arc_lengths_dense)

    # Apply moving average to smooth the points further
    x_smooth = moving_average_extended(x_smooth, smooth_window)
    y_smooth = moving_average_extended(y_smooth, smooth_window)

    # Combine the x and y dense points into an array
    smoothed_points = np.column_stack((x_smooth, y_smooth))

    return spline_function_x, spline_function_y, smoothed_points


#--------------------------------------------
#--------------------------------------------

def equidistant_points_on_curve(lon_curve, lat_curve, distance=20):
    '''
    Calculate points along a curve that are approximately equidistant based on a specified distance.

    Parameters:
    lon_curve (np.array): The longitude coordinates of the curve.
    lat_curve (np.array): The latitude coordinates of the curve.
    distance (float): Target distance in meters for equidistant points.

    Returns:
    tuple: Two arrays containing the longitude and latitude coordinates of the equidistant points along the curve.
    '''
    # Convert differences in longitude and latitude to distances in meters
    avg_lat = np.mean(lat_curve)
    lat_to_meters = 111000
    lon_to_meters = 111000 * np.cos(np.radians(avg_lat))

    # Calculate differential lengths along the curve in meters
    dx_meters = np.diff(lon_curve) * lon_to_meters
    dy_meters = np.diff(lat_curve) * lat_to_meters
    distances_meters = np.sqrt(dx_meters**2 + dy_meters**2)
    cumulative_distance = np.cumsum(distances_meters)
    cumulative_distance = np.insert(cumulative_distance, 0, 0)  # Insert 0 at the start
    
    # Initialize output arrays
    lon_equidistant = []
    lat_equidistant = []

    # Iterate through multiples of the distance
    target = distance
    for i, d in enumerate(cumulative_distance):
        if d >= target:
            lon_equidistant.append(lon_curve[i])
            lat_equidistant.append(lat_curve[i])
            target += distance  # Move to the next target distance

    return np.array(lon_equidistant), np.array(lat_equidistant)

#--------------------------------------------
#--------------------------------------------
def elevation_function(bathy, lon, lat):
    """
    Interpolates the elevation for the specified coordinates (lon, lat).

    Parameters:
    - bathy: xarray.DataArray, bathymetric dataset.
    - lon: float, longitude.
    - lat: float, latitude.

    Returns:
    - float, interpolated elevation value.
    """
    elevation = bathy.interp(lon=lon, lat=lat)
    return elevation.values.item()

#--------------------------------------------
#--------------------------------------------
def perpendicular_line(x_equidistant, y_equidistant, smoothed_points, extension_length, bathy, dummy_length = 100):
    """
    Generates perpendicular lines to the shoreline using equidistant points from the original shoreline
    and the slopes of the smoothed shoreline. Uses a dummy distance to determine the elevation direction.

    Parameters:
    - x_equidistant: numpy.ndarray, x-coordinates (longitude) of the equidistant points.
    - y_equidistant: numpy.ndarray, y-coordinates (latitude) of the equidistant points.
    - smoothed_points: numpy.ndarray, smoothed shoreline points (lon, lat).
    - extension_length: float, length of the perpendicular lines (in meters).
    - bathy: xarray.DataArray, bathymetric dataset with dimensions (lon, lat).

    Returns:
    - lines_latlon: numpy.ndarray, starting points of the perpendicular lines (lon, lat).
    - lines_UTM: numpy.ndarray, ending points of the perpendicular lines (x, y).
    """
    # Conversion factors: degrees to meters
    avg_lat = np.mean(y_equidistant)
    lat_to_meters = 111000  # Approx. 111 km per degree of latitude
    lon_to_meters = 111000 * np.cos(np.radians(avg_lat))  # Adjusted by average latitude

    # Convert smoothed_points to meters
    smoothed_points_meters = np.array([
        [lon * lon_to_meters, lat * lat_to_meters] for lon, lat in smoothed_points
    ])

    # Convert equidistant points to meters
    x_meters = x_equidistant * lon_to_meters
    y_meters = y_equidistant * lat_to_meters

    lines_latlon = []
    lines_UTM    = []
    
    for x_meter, y_meter in zip(x_meters, y_meters):
        # Find the closest point on the smoothed shoreline
        distances = np.sqrt((smoothed_points_meters[:, 0] - x_meter) ** 2 +
                            (smoothed_points_meters[:, 1] - y_meter) ** 2)
        nearest_index = np.argmin(distances)

        # Calculate the slope at the nearest point using its neighbors
        if nearest_index == 0:  # Start of the smoothed shoreline
            x_next, y_next = smoothed_points_meters[nearest_index + 1]
            x_prev, y_prev = x_meter, y_meter
        elif nearest_index == len(smoothed_points_meters) - 1:  # End of the smoothed shoreline
            x_prev, y_prev = smoothed_points_meters[nearest_index - 1]
            x_next, y_next = x_meter, y_meter
        else:  # Intermediate point
            x_prev, y_prev = smoothed_points_meters[nearest_index - 1]
            x_next, y_next = smoothed_points_meters[nearest_index + 1]

        slope = (y_next - y_prev) / (x_next - x_prev) if x_next != x_prev else np.inf
        perpendicular_slope = -1 / slope if slope != 0 else np.inf

        # while loop to avoid nan values for elevations which are produced by domain extent.
        max_attempts      = 10
        attempt           = 0
        min_dummy_length  = 100  # Define a reasonable minimum to avoid infinite loop
        dummy_length_iter = dummy_length

        while attempt < max_attempts and dummy_length_iter > min_dummy_length:
            # Calculate the dummy line endpoints
            if perpendicular_slope == np.inf:  # Vertical line
                line_end1 = (x_meter, y_meter + dummy_length_iter / 2)
                line_end2 = (x_meter, y_meter - dummy_length_iter / 2)
            else:
                delta_x_dummy = dummy_length_iter / np.sqrt(1 + perpendicular_slope**2)
                delta_y_dummy = perpendicular_slope * delta_x_dummy
                line_end1 = (x_meter + delta_x_dummy, y_meter + delta_y_dummy)
                line_end2 = (x_meter - delta_x_dummy, y_meter - delta_y_dummy)

            # Convert dummy line endpoints to lon/lat
            line_end1_lonlat = (line_end1[0] / lon_to_meters, line_end1[1] / lat_to_meters)
            line_end2_lonlat = (line_end2[0] / lon_to_meters, line_end2[1] / lat_to_meters)

            # Evaluate elevation
            elevation1 = elevation_function(bathy, *line_end1_lonlat)
            elevation2 = elevation_function(bathy, *line_end2_lonlat)

            # Check for valid elevations
            if not (np.isnan(elevation1) or np.isnan(elevation2)):
                break  # Success!

            # If invalid, reduce dummy length and retry
            dummy_length_iter *= 0.9
            attempt += 1

        # Choose the correct direction based on elevation
        if elevation1 > elevation2:
            direction = 1  # Use line_end1 as the direction
        else:
            direction = -1  # Use line_end2 as the direction

        # Calculate the final line endpoints with extension_length
        if perpendicular_slope == np.inf:  # Vertical line
            line_start = (x_meter, y_meter)
            line_end = (x_meter, y_meter + direction * extension_length / 2)
        else:
            delta_x = direction * extension_length / np.sqrt(1 + perpendicular_slope**2)
            delta_y = perpendicular_slope * delta_x
            line_start = (x_meter, y_meter)
            line_end = (x_meter + delta_x, y_meter + delta_y)

        line = np.linspace(line_start, line_end, extension_length)
        lines_latlon.append((line[:,0] /lon_to_meters, line[:,1]/lat_to_meters ))
        lines_UTM.append((line[:,0], line[:,1]))

    return lines_latlon, lines_UTM


#-------------------------------------------------------
#-------------------------------------------------------
def transect_processing(lines_latlon, lines_UTM, bathy, elevation_threshold=50):
    """
    Processes transect lines and stores their data in a dictionary of DataFrames.

    Parameters:
    - lines_latlon: list of tuples, each containing arrays of longitude and latitude points for the transects.
    - lines_UTM: list of tuples, each containing arrays of UTM X and Y points for the transects.
    - bathy: xarray.DataArray, the bathymetry data used for elevation interpolation.
    - elevation_threshold: float, elevation value above which the transect is truncated.

    Returns:
    - dict: Dictionary containing DataFrames of transect data, indexed by keys like 'T001', 'T002', etc.
    """
    # Prepare all lon and lat points from the list of lines
    all_lon = np.concatenate([line[0] for line in lines_latlon])
    all_lat = np.concatenate([line[1] for line in lines_latlon])

    # Perform interpolation for all points at once
    bathy_values = bathy.interp(lon=("points", all_lon), lat=("points", all_lat))

    # Split the bathy values back into the structure of the original list of lines
    split_indices = np.cumsum([len(line[0]) for line in lines_latlon])[:-1]
    bathy_lines = np.split(bathy_values.values, split_indices)

    # Initialize an empty dictionary to store dataframes
    lines_dataframes = {}

    # Iterate through each line and its corresponding UTM and bathy values
    for i, ((lon, lat), (x_utm, y_utm), elevation) in enumerate(zip(lines_latlon, lines_UTM, bathy_lines)):
        # Create a dataframe for the current line
        df = pd.DataFrame({
            'lon': lon,
            'lat': lat,
            'utm_X': x_utm,
            'utm_Y': y_utm,
            'elevation': elevation
        })

        # Find the first index where elevation is equal to or greater than elevation threshold
        index_threshold = df[df['elevation'] >= elevation_threshold].index.min()

        # Filter rows where elevation is within the valid range
        if pd.notna(index_threshold):
            df_filtered = df[(df['elevation'] >= 0) & (df['elevation'] <= elevation_threshold) & (df.index <= index_threshold)]
        else:
            df_filtered = df[(df['elevation'] >= 0) & (df['elevation'] <= elevation_threshold)]
        
        # Reset index after filtering and Add the filtered DataFrame to the dictionary
        df_filtered = df_filtered.reset_index(drop=True)
        lines_dataframes[f'T{str(i + 1).zfill(3)}'] = df_filtered

    return lines_dataframes

#-------------------------------------------------------
#-------------------------------------------------------

def manual_shoreline_definition(
    grid_lon, grid_lat, bathy, current_shoreline,
    n_point=1000, smooth_window=20, alpha=1
):
    """
    Allows the user to manually define a shoreline by clicking on points
    and creates a smoothed shoreline using cubic spline interpolation
    with additional options for smoothing.
    """
    # --- Define contour levels (min/max definen el clip) ---
    LEVELS = [-100, -50, -30, -20, -10, 0, 5, 10, 20, 30, 50, 100]
    i0     = LEVELS.index(0)
    k_neg  = i0
    k_pos  = len(LEVELS) - 1 - i0

    # --- Discrete colormap: GnBu para bathy, YlOrRd para topo ---
    bathy_cmap = plt.cm.GnBu
    topo_cmap  = plt.cm.gist_gray_r

    blues  = bathy_cmap(np.linspace(0.2, 1, k_neg)) if k_neg > 0 else np.empty((0, 4))
    greens = topo_cmap (np.linspace(0.2, 1, k_pos)) if k_pos > 0 else np.empty((0, 4))
    colors = np.vstack([blues, greens])
    listed = mcolors.ListedColormap(colors, name="bathy_topo_steps")

    # Claves para valores fuera de rango:
    listed.set_under(colors[0])      # < min(LEVELS) usa primer color
    listed.set_over(colors[-1])      # > max(LEVELS) usa último color

    # BoundaryNorm: bins discretos; clip=True hace que norm “pegue” a los extremos
    norm = mcolors.BoundaryNorm(LEVELS, ncolors=listed.N, clip=True)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rc('font', size=14)

    cf = ax.contourf(
        grid_lon, grid_lat, bathy,
        levels=LEVELS,
        cmap=listed,
        norm=norm,
        alpha=alpha,
        extend='both'               # pinta también <min y >max con under/over
    )

    cs = ax.contour(
        grid_lon, grid_lat, bathy,
        levels=LEVELS, colors='black', linewidths=0.5
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")

    # Colorbar del mismo alto, ticks fijos, y extensión rectangular
    cbar = fig.colorbar(cf, ax=ax, extend='both', extendrect=True, shrink=0.8)
    cbar.set_label("Topobathymetry [m]")
    cbar.locator = ticker.FixedLocator(LEVELS)
    cbar.update_ticks()

    ax.set_title("Click to define the shoreline (Right-click to Finish)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect('equal')

    if current_shoreline is not None and len(current_shoreline) > 1:
        ax.plot(current_shoreline[:, 0], current_shoreline[:, 1],
                'k-', lw=2, label='Current Shoreline')
        ax.legend(loc="lower left")

    fig.tight_layout()
    plt.draw()

    points = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig)

    if len(points) < 2:
        raise ValueError("At least two points are required to define a shoreline.")

    user_defined_shoreline = np.array(points)

    _, _, smoothed_points = create_spline_function(
        shoreline=user_defined_shoreline,
        n_point=n_point,
        smooth_window=smooth_window
    )

    return smoothed_points, user_defined_shoreline

#-------------------------------------------------------
#-------------------------------------------------------
def plot_transect_elevations(transect_dict, outfigPath):
    """
    Plots the elevation profiles for each transect in two panels:
    - Full transect profile
    - Zoomed-in profile up to 50 m

    Adds matching colorbars for both subplots to indicate transect IDs.

    Parameters:
    - transect_dict: dict, Dictionary of transects where each value is a DataFrame with 'Elevation'.
    - outfigPath: pathlib.Path object, where the figure will be saved.

    Returns:
    - None (saves and displays the plot).
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Generate a colormap and normalization for transects
    num_transects = len(transect_dict)
    cmap = plt.get_cmap("magma")
    cmap = truncate_colormap(cmap, 0, 0.7)
    norm = mcolors.Normalize(vmin=0, vmax=num_transects - 1)

    maxlen = []
    for i, (transect_name, df) in enumerate(transect_dict.items()):
        color = cmap(norm(i))
        maxlen.append(len(df))
        distance = np.linspace(0, len(df) - 1, len(df))

        ax[0].plot(distance, df["elevation"], linestyle="-", linewidth=1.5, color=color, label=transect_name)
        ax[1].plot(distance[:51], df["elevation"].iloc[:51], linestyle="-", linewidth=1.5, color=color)

    # Set titles and labels
    ax[0].set_ylabel("Elevation [m]")
    ax[0].set_title("Elevation Profiles for Transects")
    ax[0].set_xlabel("Distance Along Transect [m]")
    ax[0].grid(True, linestyle="--", alpha=0.6)

    ax[1].set_xlabel("Distance Along Transect [m]")
    ax[1].set_ylabel("Elevation [m]")
    ax[1].set_title("Elevation up to 50 m")
    ax[1].grid(True, linestyle="--", alpha=0.6)

    # Create a shared ScalarMappable for both colorbars
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    transect_keys = list(transect_dict.keys())
    tick_positions = np.linspace(0, num_transects - 1, min(9, num_transects)).astype(int)
    tick_labels = [transect_keys[i] for i in tick_positions]

    # Colorbar for ax[0]
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="3%", pad=0.15)
    cbar0 = fig.colorbar(sm, cax=cax0, ticks=tick_positions)
    cbar0.ax.set_yticklabels(tick_labels)
    cbar0.set_label("Transect ID")

    # Colorbar for ax[1]
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="3%", pad=0.15)
    cbar1 = fig.colorbar(sm, cax=cax1, ticks=tick_positions)
    cbar1.ax.set_yticklabels(tick_labels)
    cbar1.set_label("Transect ID")

    # Final layout adjustments
    fig.subplots_adjust(hspace=0.25)
    plt.tight_layout()
    plt.savefig(outfigPath / f"Transect_elevation.png", dpi=300, bbox_inches='tight', format='png')

    plt.show(block=False)

#-------------------------------------------------------
#-------------------------------------------------------
def plot_slope_distribution(transect_dict, outfigPath):
    '''
    Plot the slope histogram

    Parameters:
    - transect_dict: dict, Dictionary of transects where each value is a DataFrame with 'Elevation'.
    '''
    allslope = []

    for transect_name, df in transect_dict.items():
        deltax = 50
        
        try:
            elev_min, elev_max = df['elevation'].iloc[0], df['elevation'].iloc[deltax]
        except:
            deltax = len(df['elevation'])
            elev_min, elev_max = df['elevation'].iloc[0], df['elevation'].iloc[deltax-1]
        slope = (elev_max - elev_min) / deltax
        allslope.append(slope)

    plt.figure(figsize=(10, 5))
    
    # Use matplotlib hist for compatibility
    plt.hist(allslope, bins=25, density=False, alpha=0.6, edgecolor='black', color='#6A5ACD')

    # Optional KDE overlay
    try:
        sns.kdeplot(allslope, color='#191970')
    except:
        pass

    # Add threshold lines
    plt.axvline(x=0.1, color='#DC143C', linestyle='--', label='Slope = 10%')
    plt.axvline(x=0.01, color='#DAA520', linestyle='--', label='Slope = 1%')
    plt.axvline(x=0.001, color='#2E8B57', linestyle='--', label='Slope = 0.1%')

    plt.xlabel('Slope [m/m]')
    plt.ylabel('Frequency [Count]')
    plt.title('Slope Distribution')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.xlim(left=0)
    plt.savefig(outfigPath / "Slope_distribution.png", dpi=300, bbox_inches='tight')
    plt.show(block=False)

#-------------------------------------------------------
#-------------------------------------------------------

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_colors = cmap(np.linspace(minval, maxval, n))
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})", new_colors
    )
#-------------------------------------------------------
#-------------------------------------------------------
def plot_transects_over_topobathy(
    grid_lon, grid_lat, bathy,
    transect_dict, shoreline, x_equidistant, y_equidistant,
    smoothed_shoreline, outfigPath
):
    '''
    Plots topobathymetry with contour lines and transects, and saves the figure.

    Parameters:
    - grid_lon, grid_lat (2D arrays): Coordinates of the grid.
    - bathy (2D array): Bathymetric elevation data.
    - transect_dict (dict): Dictionary of transects with keys and 'Lon'/'Lat' as values.
    - shoreline (2D array): Coordinates of the z=0 m contour.
    - x_equidistant, y_equidistant (arrays): Scatter points along smoothed shoreline.
    - smoothed_shoreline (2D array): User-defined shoreline coordinates.
    - city (str): Name of the city for naming output folder.
    - cmap_bathy (str): Colormap to use for bathymetry background.
    '''

    # Create a single subplot for the map
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))

    # Determine elevation levels for contour plotting
    elev_min = round(bathy.min() / 10) * 10
    elev_max = round(bathy.max() / 10) * 10
    elev_delta = round((elev_max - elev_min) / 11)

    # Plot topobathymetry and extract the contour fill object for colorbar
    axs, cp = plot_topobathymetry_and_contours(
        grid_lon, grid_lat, bathy,
        z0_contour=shoreline,
        ax=axs,
        alpha=0.6
    )

    # Define a colormap for transects
    num_transects = len(transect_dict)
    cmap = plt.get_cmap("magma")
    cmap = truncate_colormap(cmap, 0, 1)
    norm = mcolors.Normalize(vmin=0, vmax=num_transects - 1)

    # Plot each transect line with a unique color
    for ix, df in enumerate(transect_dict.values()):
        line_lon = df['lon']
        line_lat = df['lat']
        line, = axs.plot(line_lon, line_lat, ls='-', lw=2, c=cmap(norm(ix)))
        line.set_path_effects([
            pe.Stroke(linewidth=1, foreground='black'),  # borde
            pe.Normal()                                  # línea original encima
        ])

    # Create a ScalarMappable for the transect colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Attach both colorbars to the right of the plot
    divider = make_axes_locatable(axs)

    # Elevation colorbar
    CUSTOM_LEVELS = [-100, -50, -30, -20, -10, 0, 5, 10, 20, 30, 50, 100]
    cax1 = divider.append_axes("right", size="5%", pad=0.1)
    cbar1 = fig.colorbar(cp, cax=cax1, ticks=CUSTOM_LEVELS)
    cbar1.set_label('Elevation [m]')

    # Transect ID colorbar
    cax2 = divider.append_axes("right", size="5%", pad=0.9)
    cbar2 = fig.colorbar(sm, cax=cax2, ticks=np.linspace(0, num_transects - 1, 9).astype(int))
    transect_keys = list(transect_dict.keys())
    cbar2.ax.set_yticklabels([transect_keys[i] for i in np.linspace(0, num_transects - 1, 9).astype(int)])
    cbar2.set_label("Transect ID")

    # Plot shoreline reference points
    axs.scatter(x_equidistant, y_equidistant, c='k')
    axs.plot(
        smoothed_shoreline[:, 0], smoothed_shoreline[:, 1],
        ls='--', lw=2, c='y', label='Shoreline defined by User'
    )

    # Set plot bounds and labels
    axs.set_xlim(grid_lon.min(), grid_lon.max())
    axs.set_ylim(grid_lat.min(), grid_lat.max())
    axs.set_aspect('equal')
    axs.legend()
    axs.set_title(f'Number of transects: {len(transect_dict)}')

    plt.tight_layout()
    plt.savefig(outfigPath / f"Transects_over_topobathy.png", dpi=300, bbox_inches='tight', format='png')
    # Show plot
    plt.show(block=False)