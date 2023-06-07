#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MAIN_DIR=$SCRIPT_DIR/../

datafolders=("3dof_lqr" "3dof_mpc" "3dof_ol" "3dof_trajectory_pointing_constraints" "6dof_lqr" "6dof_trajectory")
figurefolders=("3dof_comparison" "3dof_trajectory_no_constraints" "3dof_vs_6dof_comparison")
figurefolders+=(${datafolders[@]})

# Create data directory
for directory in ${datafolders[@]}; do 
    mkdir -p $MAIN_DIR/data/$directory
done

# Create figure directory
for directory in ${figurefolders[@]}; do 
    mkdir -p $MAIN_DIR/figures/$directory
done