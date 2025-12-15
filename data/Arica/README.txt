Toolkit requirements

The topo-bathymetric data must be provided as a NetCDF file containing two coordinate vectors, typically latitude and longitude, and a 2D matrix, with positive values representing topography and negative values representing bathymetry.

Tsunami simulations must also be provided as NetCDF files. Files must contain only the maximum recorded water depth fields, sobered as a 3D array with two spatial dimensiones for the simulation grid and a third dimension indexing individual scenarios.

Both NetCDF files must be spatially aligned.