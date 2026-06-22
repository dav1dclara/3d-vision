# Quality Assessment

This document describes how to run the mesh quality assessment GUI and the required dependencies.

## Overview
The quality assessment tool compares a reconstructed mesh against a reference point cloud and reports distance metrics and residual distribution. It also provides an optional Plotly visualization.

## Requirements
See the top-level `requirements.txt` for the dependency list.

## Run
1) Activate your Python environment.
2) Install dependencies: `pip install -r requirements.txt`.
3) Start the GUI:

python scripts/run_quality_assessment.py

## Usage
1) Click Browse next to Mesh (.ply) and select the reconstructed mesh.
2) Click Browse next to Point Cloud (.ply) and select the reference point cloud.
3) Set Max sample size (or enable Use all points).
4) Adjust thresholds (cm) to define Good / OK / Critical / Missing ranges.
5) Select metric categories with the checkboxes.
6) Click Evaluate to run the assessment.
7) Use Open Plot to view the visualization if enabled.

## Demo
Video walkthrough:
https://n-joy-nas.quickconnect.to/d/s/18InB7RiatQochLqLGiQHq3WTJrcynmf/kbzfSNGYkPu70ODU3X0ZMR4IGRrTIiBm-57EAGZYdNQ0

### Options
- Max sample size: Upper bound for point sampling used by distance and residual metrics.
- Use all points: Disables subsampling and uses the full point cloud.
- Thresholds (cm): Defines the residual distribution bins.
- Metrics:
	- Structure: Mesh/point cloud counts and mesh triangle stats.
	- Distance: RMSE/MAE/Hausdorff and residual stats.
	- Residual Distribution: Percentages in each threshold range.
	- F-Score: Optional overlap metric (can be slow on large inputs).
	- Visualization: Plotly colored residual plot.
