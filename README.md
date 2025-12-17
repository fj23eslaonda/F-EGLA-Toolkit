# <span style="font-size:34px; font-style:italic;">F-EGLA: A Python Toolkit for Rapid Estimation of Tsunami-Induced Flooded Areas</span>

### <span style="font-size:14px; font-weight:600;">Authors:</span>

<span style="font-size:16px;">
Francisco J. Sáez<sup>a</sup>,  
Rodrigo Cienfuegos<sup>b</sup>,  
Patricio A. Catalán<sup>c</sup>,  
Alejandro Urrutia<sup>a</sup>
</span>

### <span style="font-size:14px; font-weight:600;">Affiliations:</span>

<div style="font-size:14px; line-height:1.4;">
a. Centro de Investigación para la Gestión Integrada del Riesgo de Desastres (CIGIDEN), Santiago, Chile  
</div>
<div style="font-size:14px; line-height:1.4;">
b. Departamento de Ingeniería Hidráulica y Ambiental, Escuela de Ingeniería, Pontificia Universidad Católica de Chile, Santiago, Chile  
</div>
<div style="font-size:14px; line-height:1.4;">
c. Departamento de Obras Civiles, Universidad Técnica Federico Santa María, Valparaíso, Chile  
</div>


---

## 🔍 Overview

**F-EGLA-Toolkit** is an open-source Python framework for **fast tsunami inundation mapping** based on the *Forward Energy Grade Line Analysis* ([FEGLA](https://www.sciencedirect.com/science/article/abs/pii/S0378383924002217)) method and the **Energy Grade Line Analysis (EGLA)** workflow for computing transect-based tsunami hydrodynamic profiles over a predefined flooded area ([EGLA from ASCE](https://ascelibrary.org/doi/book/10.1061/9780784415788)).

The **FEGLA-Toolkit** provides a physically consistent surrogate to full **Nonlinear Shallow Water Equation (NSWE)** simulations and enables users to:

- Estimate tsunami-induced inundated areas for multiple scenarios   
- Evaluate model performance against ground-truth NSWE simulations  
- Run a complete, reproducible hazard-mapping pipeline in Python
- Generate inundation polygons (KMZ/SHP) for GIS visualization

The **EGLA-Toolkit** is used to compute transect-based tsunami hydrodynamic profiles over a predefined flooded area. Setting user-defined transects and performing the following steps:

- Runs **EGLA** along each transect to compute flow depth, velocity, and Froude number using prescribed parameters (F0, FR, Manning’s n).
- Integrates the energy grade line **from the runup point toward the shoreline**, accounting for terrain elevation and friction losses.
- Reports diagnostic hydrodynamic values at the shoreline for each transect.
- Generates **hydrodynamic profile plots** (flow depth, velocity, and Froude number) along cumulative distance.
- Saves the computed transect-by-transect results as CSV files for further analysis and reproducibility.

---
## Repository Structure
```
F-EGLA-Toolkit/
│
├── data/
│   └── <city>/                # Contains bathymetry and SWE simulations in NetCDF format
|
├── tsunamicore/
│   ├── fegla/
│   │   ├── __init__.py
│   │   ├── model.py           # Core FEGLA algorithm
│   │   └── operations.py      # Operations to help FEGLA
│   │
│   ├── postprocessing/
│   │   ├── __init__.py
│   │   ├── notebook_tools.py  # Contains functions to execute in notebook
│   │   └── results.py         # Contains functions to process the calibration results
│   │
│   ├── preprocessing/          
│   │   ├── __init__.py
│   │   ├── transects.py       # Functions to create and manage transect
│   │   └── simulation.py      # Function to manage SWE simulations
│   │
│   └── utils/
│       ├── __init__.py
│       └── plot_style.py      # Global styles for Matplotlib
│
├── scripts/
│   ├── config/
|       ├── City_inputs.json   # Parameters to calibrate F-EGLA in a certain city
│   ├── calibrate_fegla.py     # Main code to calibrate FEGLA
│   ├── get_hmax.py            # Obtain and manage hmax from SWE simulation
│   ├── get_transects_FEGLA.py # Obtain and manage transects for FEGLA
│   ├── get_transects_EGLA.py  # Obtain and manage transects for EGLA
│   └── area_results.py        # Processing area results
│
├── outputs/
│   └── City/                  # All outputs of the F-EGLA application for a certain city are saved here.
│       └── README.txt
│
├── notebook/
│   └── FEGLA_run.ipynb        # Notebook for friendly FEGLA application
│   └── EGLA_run.ipynb         # Notebook for friendly EGLA application
│
├── venv/                      # Virtual environment for this implementation
│
├── README.md
├── requirements.txt           # Python packages list to install
├── pyproject.toml             # Define how a Python project is built
└── LICENSE
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/<YOUR_USERNAME>/F-EGLA-Toolkit.git
cd F-EGLA-Toolkit
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Install the project locally
```bash
pip install -e .
```

---

## **FEGLA - Step-by-Step Application**

### **Step 1: data preprocessing**
Ensure the required datasets (bathymetric profiles, tsunami scenarios) are available in the `FEGLA-Toolkit/Data/YOUR_CITY`.
  - The folder related to a specific city must be created inside the Data folder.
  - To standardize the data format, the netCDF format is used due to the nature of the data (lon, lat, h).
  - Topobathymetry data must have the following format, and it must be named as: “Bathymetry.nc”.
  - Each simulation file must have the following format and its name must start with “hmax”.

### **Step 2: Create transects**
This step extracts transects for a specified city using the `get_transects_FEGLA.py` script.

```bash
python scripts/get_transects_FEGLA.py --city YOUR_CITY --extension_length 5000 --distance 100 --elevation_threshold 30
```
Parameter Definitions:

	•	--city → Name of the city folder (e.g., "Arica").
	•	--extension_length → Initial length of each transect in meters (e.g., 5000).
	•	--distance → Spacing between consecutive transects (typically 50m or 100m).
	•	--elevation_threshold → Maximum elevation used as a threshold for transects.

Once executed, a bathymetry plot is displayed. The user must click to define the shoreline, which finalizes the transect extraction process.

### **Step 3: The user must decide whether to calibrate the model**

If the user wants to apply the FEGLA method without calibration, the user must go to `FEGLA-Toolkit/notebook/FEGLA_run.ipynb`.

The recommendation is to calibrate the method since it is site-dependent. Then, the user must go to Step 3.

### **Step 4: Obtain flooded transects**
After generating the transect data, the next step is to **interpolate the transects** across all inundation maps derived from the Shallow Water Equations (SWE).

```bash
# Ensure you are in the Processing directory:
# Run the interpolation script
python scripts/get_hmax.py --city YOUR_CITY --n_selected_sim 50
```

Parameter Definitions:
	•	--city → Name of the city folder (e.g., "Arica").
	•	--n_selected_sim → Number of selected simulations from the total available SWE simulations.

For a given location, thousands of SWE simulations (e.g., 3000) may be available. This script selects n_selected_sim simulations based on mean flooded heights at the shoreline, a key parameter for FEGLA. 

### **Step 5: Executing the Models**
To determine the best-fit model, three different parameterizations of the Froude number are tested: **Constant, Squared, and Linear**.

1. **Prepare the JSON configuration file** (`params_inputs_city.json`), which contains the required inputs:

   Example: `params_inputs_Arica.json`
   ```json
   {
     "city": "Arica",
     "batch_size": 12,
     "manning": 0.04,
     "selected_scenarios": "Selected_scenarios_Nsim_50.pkl",
     "F0": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
     "FR": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
   }

2. **Run the FEGLA model for different Forude number parameterizations**
```bash
python script/calibrate_fegla.py --params Arica_inputs.json
```
3.	**Results Storage**
	•	All outputs are saved in the `outputs/YOUR_CITY/calibration` directory 
	•	The results are stored as .pkl files for further analysis.

### **Step 6: Evaluating the Best-Fit Model**
After executing all simulations, the next step is to identify the **best-fit FEGLA model** by comparing its results against the **Shallow Water Equations (SWE) simulations**, which serve as a benchmark.

```bash
python scripts/area_results.py --city Arica --n_selected_sim 50 --map_format kmz
```

### **Step 7: Testing the Best-Fit Model**
The user must go to `FEGLA-Toolkit/notebook/FEGLA_run.ipynb` and set all parameters related to the best-fit model

---

## **EGLA - Step-by-Step Application**

### **Step 1: data preprocessing**
Ensure the required dataset (bathymetric profiles) is available in the `F-EGLA-Toolkit/Data/YOUR_CITY`.
  - The folder related to a specific city must be created inside the Data folder.
  - To standardize the data format, the netCDF format is used due to the nature of the data (lon, lat, h).
  - Topobathymetry data must have the following format, and it must be named as: “Bathymetry.nc”.

### **Step 2: Create transects**
The script get_transects_EGLA.py is used to define and generate the transects required by the EGLA workflow. It provides an interactive environment where the user creates transects directly over topobathymetry and a flooded-area map.

```bash
python scripts/get_transects_EGLA.py --city YOUR_CITY
```
Parameter Definitions:

	•	--city → Name of the city folder (e.g., "Arica").

When the script is executed, an interactive map is displayed showing the topobathymetry together with the flooded-area polygon derived from a KMZ or SHP file. The user defines each transect by clicking two points within the flooded area to specify its orientation and pressing Enter to confirm its creation. The resulting transect is immediately projected and displayed on the map, allowing visual validation before proceeding. This process can be repeated to define multiple transects. Once all desired transects have been created, the user confirms the process to finalize and save the transect data for subsequent EGLA analysis.

### **Step 3: Running EGLA**

The user must go to `F-EGLA-Toolkit/notebook/EGLA_run.ipynb`, set the parameters that the user prefers, and the EGLA method will be applied over the user-defined transects.

## Authors

* **Francisco Sáez R.** - Assistant Researcher, CIGIDEN - [fj23eslaonda](https://github.com/fj23eslaonda)

## Acknowledgments

* [CIGIDEN](https://www.cigiden.cl/en/home/)
