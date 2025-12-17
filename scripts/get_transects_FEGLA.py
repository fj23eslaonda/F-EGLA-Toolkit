#--------------------------------------------------------
# PACKAGES
#--------------------------------------------------------
import matplotlib.pyplot as plt
import xarray as xr
import argparse
import pandas as pd

from pathlib import Path
from tsunamicore.preprocessing.transects import (
    manual_shoreline_definition,
    equidistant_points_on_curve,
    perpendicular_line,
    transect_processing,
    plot_transects_over_topobathy,
    find_contour_coordinates,
    plot_transect_elevations,
    plot_slope_distribution
)

#--------------------------------------------------------
# Main function
#--------------------------------------------------------

def main(city, extension_length, distance, elevation_threshold):
    """
    Generates shoreline, and transects from bathymetry data and saves them
    to the project's data directory.

    Parameters
    ----------
    city : str
        City name, used to locate bathymetry files.
    extension_length : int
        Transect length in meters.
    distance : int
        Spacing between equidistant points along shoreline.
    elevation_threshold : float
        Threshold for filtering transect elevations.
    """
    root        = Path(__file__).resolve().parent.parent
    data_path   = root / 'data' / city

    # Creating output folder
    output_path = root / 'outputs' / city
    output_path.mkdir(parents=True, exist_ok=True)

    # Creating fig folder
    outfig_path = output_path / 'figs'
    outfig_path.mkdir(parents=True, exist_ok=True)

    # Creating transect folder
    transect_path = output_path / 'transects'
    transect_path.mkdir(parents=True, exist_ok=True)

    while True:
        # Load bathymetry dataset
        bathy_nc = xr.open_dataset(data_path / 'Bathymetry.nc')
        grid_lat = bathy_nc["lat"].values
        grid_lon = bathy_nc["lon"].values
        bathy    = bathy_nc["bathy"].values

        # Find shoreline
        shoreline = find_contour_coordinates(grid_lon, grid_lat, bathy, level=0)
        # Remove values from the ends of the line
        shoreline = shoreline[10:-10]

        # User defines the shoreline
        smoothed_shoreline, _ = manual_shoreline_definition(
            grid_lon, grid_lat, bathy, shoreline, n_point=1000, smooth_window=20, alpha=0.6
        )

        # Compute equidistant points along the shoreline
        x_equidistant, y_equidistant = equidistant_points_on_curve(
            smoothed_shoreline[:, 0], smoothed_shoreline[:, 1], distance
        )

        # Generate perpendicular transects
        lines_latlon, lines_UTM = perpendicular_line(
            x_equidistant,
            y_equidistant,
            smoothed_shoreline,
            extension_length = extension_length,
            bathy            = bathy_nc['bathy'],
            dummy_length     = 500
        )
        
        # Process transects into dictionary
        transect_dict = transect_processing(
            lines_latlon, lines_UTM, bathy_nc["bathy"], elevation_threshold
        )

        # Plot Figures
        plot_transects_over_topobathy(grid_lon, grid_lat, bathy,
                                      transect_dict, shoreline,
                                      x_equidistant, y_equidistant,
                                      smoothed_shoreline, outfig_path
        ) 

        #Plot the elevation profiles for transects
        plot_transect_elevations(transect_dict, outfig_path)

        # Plot slope distribution
        plot_slope_distribution(transect_dict, outfig_path)

        # Ask user whether to continue or exit
        user_input = input("\nProceed with saving these results? (y/n): ").strip().lower()
        if user_input == 'y':
            # Save transects as HDF5 file
            store_path = transect_path / 'FEGLA_transects_data.h5'
            with pd.HDFStore(store_path, mode='w') as store:
                for key, df in transect_dict.items():
                    store[key] = df
            print("\nFinalizing process...")
            break
        else:
            plt.close('all')
            print("\nRepeating process with new inputs...\n")

#--------------------------------------------------------
# RUN
#--------------------------------------------------------

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Generate shoreline transects from bathymetry data.")
    parser.add_argument("--city", type=str, required=True, help="City to create the Path to the bathymetry NetCDF file")
    parser.add_argument("--extension_length", type=int, required=True, help="Length of the transects in meters")
    parser.add_argument("--distance", type=int, required=True, help="Distance between equidistant points")
    parser.add_argument("--elevation_threshold", type=int, required=True, help="Elevation threshold for filtering transects")

    # Parse arguments
    args = parser.parse_args()

    # Run main function
    main(args.city, args.extension_length, args.distance, args.elevation_threshold)