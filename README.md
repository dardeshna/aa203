# Optimal Control for Minimum-Fuel Pinpoint Landing
## AA 203: Optimal & Learning Based Control
Devin Ardeshna, Nick Delurgio, and Pol Francesch Huc

---

## Introduction

This repository encompasses the work the team conducted as part of AA 203 at Stanford Spring 2023. A full report can be found in this repository.

## Environment Setup
This environment setup assumes you have installed Python (version >= 3.8) on your machine. 
```
python -m venv aa203_final_project
```
Load the environment

Windows:
```
aa203_final_project\Scripts\activate
```
MacOS/Linux:
```
source aa203_final_project/bin/activate
```

Installing libaries:
```
pip install -r ./requirements.txt
```

Create directory structure:
```
bash ./scripts/create_dir_structure.sh
```

## Running the code
First load the environment as before, and then run the following to generate all the relevant 3DOF data and plots:
```
bash ./scripts/run_3dof.sh
python ./plot_scripts/compare_3dof.py
```

To generate the 6DOF plots you can run:
```
python ./6dof/freefinaltime.py
python ./6dof/lqr_tracking.py

python ./plot_scripts/compare_trajectories.py
```