# <span style="font-size:34px; font-style:italic;">FEGLA: A Python Toolkit for Rapid Estimation of Tsunami-Induced Flooded Areas</span>

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

**FEGLA-Toolkit** is an open-source Python framework for **fast tsunami inundation mapping** based on the *Forward Energy Grade Line Analysis* ([FEGLA](https://www.sciencedirect.com/science/article/abs/pii/S0378383924002217)) method 

The toolkit provides a physically consistent surrogate to full **Nonlinear Shallow Water Equation (NSWE)** simulations and enables users to:

- Estimate tsunami-induced inundated areas for multiple scenarios   
- Evaluate model performance against ground-truth NSWE simulations  
- Run a complete, reproducible hazard-mapping pipeline in Python
- Generate inundation polygons (KMZ/SHP) for GIS visualization

---
## Repository Structure
```
FEGLA-Toolkit/
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
|       ├── City_inputs.json   # Parameters to calibrate FEGLA in a certain city
│   ├── calibrate_fegla.py     # Main code to calibrate FEGLA
│   ├── get_hmax.py            # Obtain and manage hmax from SWE simulation
│   ├── get_transects.py       # Obtain and manage transects
│   └── area_results.py        # Processing area results
│
├── outputs/
│   └── City/                  # All outputs of the FEGLA application for a certain city are saved here.
│       └── README.txt
│
├── notebook/
│   └── FEGLA_run.ipynb        # Notebook for friendly FEGLA application
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
git clone https://github.com/<YOUR_USERNAME>/FEGLA-Toolkit.git
cd FEGLA-Toolkit
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

## FEGLA Application
### 1. Obtaining transects using topo-bathymetry
```bash
 python scripts/get_transects.py --city Arica --extension_length 5000 --distance 100 --elevation_threshold 50
```
