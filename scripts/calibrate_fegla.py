# ==============================================================================
# Main file to execute FEGLA_runner.py
#
# File Name    : Main_FEGLA.py
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
import argparse
import json
from pathlib import Path
from tsunamicore.fegla.model import FEGLA_calibration

#--------------------------------------------------------
# Loading input file
#--------------------------------------------------------
def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

#--------------------------------------------------------
# FEGLA Execution
#--------------------------------------------------------
if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    config_path = root / 'config'

    parser = argparse.ArgumentParser(description="Run FEGLA model or calibration.")
    parser.add_argument('--params', type=str, required=True, help="Path to JSON parameter file.")
    args = parser.parse_args()
    json_path = config_path / args.params

    params = load_json(json_path)
    FEGLA_calibration(params) 