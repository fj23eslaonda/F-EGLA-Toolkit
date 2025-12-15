#--------------------------------------------------------
#
# PACKAGES
#
#--------------------------------------------------------
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
 
from tsunamicore.postprocessing.results import (
    extract_cte_squared_name,
    extract_linear_name,
    plot_1Derror_distribution,
    plot_2Derror_distribution,
    load_flooded_ground_truth,
    load_areas_for_mode,
    compute_config_area_errors_and_bias,
    load_best_transects_and_areas,
    load_hmax_to_curve,
    selecting_scenario_for_bathub,
    compute_bathtub_area_for_list,
    plot_flooding_curve,
    find_min_values,
    save_polygons_by_scenario
)

#--------------------------------------------------------
#
# Main function
#
#--------------------------------------------------------
def main(city, Nsim, mapformat):

    root = Path(__file__).resolve().parent.parent
    data_path        = root / 'data' / city
    output_path      = root / 'outputs' / city
    metadata_path    = output_path / 'metadata' 
    inundation_path  = output_path / 'inundation'
    calibration_path = output_path / 'calibration'

    outfigs_path = calibration_path / 'figs'
    outfigs_path.mkdir(parents=True, exist_ok=True)

    outkml_path = calibration_path / 'kml'
    outkml_path.mkdir(parents=True, exist_ok=True)
    
    outshp_path = calibration_path / 'shp'
    outshp_path.mkdir(parents=True, exist_ok=True)

    print(' ------------------------')
    print(' Loading data and computing areas ')
    print(' ------------------------\n')

    ground_truth_areas, ground_truth_polygon, transect_geometry = load_flooded_ground_truth(city = city, Nsim = Nsim, loading = 'calibration', 
                                                                                            metadata_root = metadata_path, inundation_root = inundation_path, column = 'hmax')
    
    print('\n')
    areas_constant = load_areas_for_mode(outputPath        = calibration_path,
                                         Nsim              = Nsim,
                                         transect_geometry = transect_geometry,
                                         mode              = "constant",
                                         name_extractor    = extract_cte_squared_name,
    )
    print('\n')
    areas_squared = load_areas_for_mode(outputPath         = calibration_path,
                                        Nsim               = Nsim,
                                        transect_geometry  = transect_geometry,
                                        mode               = "squared",
                                        name_extractor     = extract_cte_squared_name,
    )
    print('\n')
    areas_linear = load_areas_for_mode(outputPath          = calibration_path,
                                       Nsim                = Nsim,
                                       transect_geometry   = transect_geometry,
                                       mode                = "linear",
                                       name_extractor      = extract_linear_name,
    )

    print(' ------------------------')
    print(' Computing errors ')
    print(' ------------------------\n')
    error_constant, bias_constant = compute_config_area_errors_and_bias(areas_constant, ground_truth_areas)
    error_squared, bias_squared   = compute_config_area_errors_and_bias(areas_squared, ground_truth_areas)
    error_linear, bias_linear     = compute_config_area_errors_and_bias(areas_linear, ground_truth_areas)


    min_F0_cte, min_err_cte, min_F0_sq, min_err_sq = plot_1Derror_distribution(error_constant, error_squared, outfigs_path, Nsim,
                                                                               bias_constant=bias_constant, bias_squared=bias_squared)

    min_F0_linear, min_FR_linear, min_err_linear  = plot_2Derror_distribution(error_data=error_linear, bias_data=bias_linear,
                                                                              outfigPath=outfigs_path, Nsim=50, plot_type="Area_",
                                                                              eb_levels=(-20, -10, 0, 10, 20)        # draws EB=0 (solid) and ±5, ±10 (dashed)
                                                                              )

    # print(' ------------------------')
    # print(' Computing Tsunami Inundation Curve ')
    # print(' ------------------------\n')

    #ground_truth_areas_all, _, transect_geometry = load_flooded_ground_truth(city = city, Nsim=None, 
    #                                                                         loading='all', metadata_root = metadata_path,
    #                                                                         inundation_root = inundation_path, column='hmax')
    
    #ground_truth_hmax_all = load_hmax_to_curve(city, Nsim, data_root=metadata_path, mode='all')

    best_type, best_vals = find_min_values(min_F0_cte, min_err_cte,
                                          min_F0_sq,  min_err_sq,
                                          min_F0_linear, min_FR_linear, min_err_linear)
    
    FEGLA_areas, FEGLA_polygon = load_best_transects_and_areas(city, Nsim, calibration_path, transect_geometry, best_type, best_vals, column='height')

    #FEGLA_hmax = load_hmax_to_curve(city, Nsim, metadata_path, mode='FEGLA')

    #bath_selectedsim = selecting_scenario_for_bathub(ground_truth_hmax_all, n_points=30)

    #Areabath = compute_bathtub_area_for_list(city, bath_selectedsim, data_root=data_path)

    #Hmeanbath = [ground_truth_hmax_all[scenario] for scenario in bath_selectedsim]

    # plot_flooding_curve(ground_truth_areas_all,
    #                     ground_truth_hmax_all,
    #                     FEGLA_areas,
    #                     FEGLA_hmax,
    #                     Areabath,
    #                     Hmeanbath,
    #                     outfigs_path
    # )

    print(' ------------------------')
    print(' Saving polygons as shp/kmz ')
    print(' ------------------------\n')

    if mapformat == 'kmz':
        format = 'kmz'
        outmap_path = outkml_path
    if mapformat == 'shp':
        format = 'shp'
        outmap_path = outshp_path

    save_polygons_by_scenario(
        city           = city,
        swe_polygons   = ground_truth_polygon,   # o ground_truth_polygon_all
        fegla_polygons = FEGLA_polygon,          # tu dict equivalente
        out_root       = outmap_path,
        fmt            = format,                  # o "shp"
        source_crs     = "EPSG:32719",       
        target_crs     = "EPSG:4326",            # Check CRS
        color_map      = {"SWE": "#ffa600", 
                          "FEGLA": "#003f5c"}
    )

    # Ask user whether to exit
    input("\nPress Enter to close all figures and finish...")
    plt.close('all')
    print("Figures closed. Process completed.")

#--------------------------------------------------------
#
# Execute code
#
#--------------------------------------------------------
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process flood simulation data.")
    parser.add_argument("--city", type=str, required=True, help="City name for data processing.")
    parser.add_argument("--n_selected_sim", type=int, required=True, help="Number of selected simulations.")
    parser.add_argument("--map_format", type=str, required = True, help = "Choose the map format.")
    args = parser.parse_args()

    # Assign parsed arguments
    city = args.city
    Nsim = args.n_selected_sim
    mapformat = args.map_format

    # Run profiler
    #profiler = cProfile.Profile()
    #profiler.enable()

    # Run main function
    main(city, Nsim, mapformat)

    #profiler.disable()

    # # Generar reporte y guardarlo en archivo
    # with open("profile_results.txt", "w") as f:
    #     stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
    #     stats.print_stats(30)  # muestra las 30 funciones más costosas

    # print("Perfil de ejecución guardado en 'profile_results.txt'")

