#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


echo -e "Creating trajectory"
python $SCRIPT_DIR/../3dof/3dof_guidance/generateTrajectory_pointingConstraints.py

echo -e "\n\nFollowing trajectory with open-loop control"
python $SCRIPT_DIR/../3dof/3dof_tracking/lqr_and_ol_tracking.py -open_loop

echo -e "\n\nFollowing trajectory with LQR control"
python $SCRIPT_DIR/../3dof/3dof_tracking/lqr_and_ol_tracking.py

echo -e "\n\nMPC + OL"
python $SCRIPT_DIR/../3dof/3dof_mpc/3dof_mpc.py -open_loop

echo -e "\n\nMPC + LQR"
python $SCRIPT_DIR/../3dof/3dof_mpc/3dof_mpc.py