#--------------------------------------------------------
# PACKAGES
#--------------------------------------------------------
import matplotlib.pyplot as plt
import xarray as xr
import argparse
import pandas as pd
import geopandas as gpd
from pathlib import Path
from tsunamicore.preprocessing.transects import (
    manual_transect_definition,
    build_transect_dictionary,
    plot_transect_elevations,
    plot_user_transects_over_topobathy
)

#--------------------------------------------------------
# Main function
#--------------------------------------------------------

def main(city):
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

        flood_gdf = gpd.read_file("S1789.kmz").set_crs(epsg=4326)

        # User defines the shoreline
        transects, _ = manual_transect_definition(grid_lon,
                                                          grid_lat,
                                                          bathy,
                                                          flood_gdf,
                                                          alpha=0.6)

        transect_dict = build_transect_dictionary(transects,
                                                      bathy_nc["bathy"],
                                                      spacing=1.0)
        
        plot_user_transects_over_topobathy(grid_lon,
                                           grid_lat,
                                           bathy,
                                           transect_dict,
                                           outfig_path,
                                           alpha=0.6)
        
        #Plot the elevation profiles for transects
        plot_transect_elevations(transect_dict, outfig_path, mode='EGLA')

        # Ask user whether to continue or exit
        user_input = input("\nProceed with saving these results? (y/n): ").strip().lower()
        if user_input == 'y':
            # Save transects as HDF5 file
            store_path = transect_path / 'EGLA_transects_data.h5'
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
    #parser.add_argument("--extension_length", type=int, required=True, help="Length of the transects in meters")
    #parser.add_argument("--distance", type=int, required=True, help="Distance between equidistant points")
    #parser.add_argument("--elevation_threshold", type=int, required=True, help="Elevation threshold for filtering transects")

    # Parse arguments
    args = parser.parse_args()

    # Run main function
    main(args.city)#, args.extension_length, args.distance, args.elevation_threshold)