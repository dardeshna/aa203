import os

import numpy as np
import matplotlib.pyplot as plt

## Save figures
file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(file_dir, '../data')
figure_dir = os.path.join(file_dir, '../figures/', '3dof_comparison')

## Load all the trajectories
traj_data_dir = os.path.join(data_dir, '3dof_trajectory_pointing_constraints')
ol_data_dir  = os.path.join(data_dir, '3dof_ol')
lqr_data_dir = os.path.join(data_dir, '3dof_lqr')
mpc_lqr_data_dir = os.path.join(data_dir, '3dof_mpc_lqr')
mpc_ol_data_dir = os.path.join(data_dir, '3dof_mpc_ol')

traj_pos = np.load(os.path.join(traj_data_dir, "pos.npy"))
ol_x     = np.load(os.path.join(ol_data_dir, "x.npy"))
lqr_x    = np.load(os.path.join(lqr_data_dir, "x.npy"))
mpc_lqr_x    = np.load(os.path.join(mpc_lqr_data_dir, "mpc_x.npy"))
mpc_ol_x    = np.load(os.path.join(mpc_ol_data_dir, "mpc_x.npy"))


ol_u     = np.load(os.path.join(ol_data_dir, "u.npy"))
lqr_u    = np.load(os.path.join(lqr_data_dir, "u.npy"))
mpc_lqr_u    = np.load(os.path.join(mpc_lqr_data_dir, "mpc_u.npy"))
mpc_ol_u    = np.load(os.path.join(mpc_ol_data_dir, "mpc_u.npy"))

mpc_t = np.load(os.path.join(mpc_lqr_data_dir, "mpc_t.npy"))

## Plot
plt.figure()
plt.subplot(3,1,1)
plt.grid(True)
plt.plot(mpc_t,ol_x[:,0]-traj_pos[:,0])
plt.plot(mpc_t,lqr_x[:,0]-traj_pos[:,0])
plt.plot(mpc_t,mpc_ol_x[:,0]-traj_pos[:,0])
plt.plot(mpc_t,mpc_lqr_x[:,0]-traj_pos[:,0])
plt.legend(["Open Loop","LQR","MPC + OL", "MPC + LQR"], loc="upper left")
plt.ylabel(r"Error $r_x$ [m]")
plt.subplot(3,1,2)
plt.grid(True)
plt.plot(mpc_t,ol_x[:,1]-traj_pos[:,1])
plt.plot(mpc_t,lqr_x[:,1]-traj_pos[:,1])
plt.plot(mpc_t,mpc_ol_x[:,1]-traj_pos[:,1])
plt.plot(mpc_t,mpc_lqr_x[:,1]-traj_pos[:,1])
plt.ylabel(r"Error $r_y$ [m]")
plt.subplot(3,1,3)
plt.grid(True)
plt.plot(mpc_t,ol_x[:,2]-traj_pos[:,2])
plt.plot(mpc_t,lqr_x[:,2]-traj_pos[:,2])
plt.plot(mpc_t,mpc_ol_x[:,2]-traj_pos[:,2])
plt.plot(mpc_t,mpc_lqr_x[:,2]-traj_pos[:,2])
plt.xlabel("t [s]")
plt.ylabel(r"Error $r_z$ [m]")
plt.savefig(os.path.join(figure_dir,"3dof_poserr.png"))

plt.figure()
plt.subplot(3,1,1)
plt.grid(True)
plt.plot(mpc_t[:-1],ol_u[:,0]/1000,'b')
plt.plot(mpc_t[:-1],lqr_u[:,0]/1000,'r',linestyle='dashed')
plt.plot(mpc_t[:-1],mpc_ol_u[:,0]/1000,'g',linestyle='dashed')
plt.plot(mpc_t[:-1],mpc_lqr_u[:,0]/1000,'k',linestyle='dashed')
plt.ylabel(r"$T_x$ [kN]")
plt.legend(["Open Loop","LQR","MPC + OL", "MPC + LQR"])

plt.subplot(3,1,2)
plt.grid(True)
plt.plot(mpc_t[:-1],ol_u[:,1]/1000,'b')
plt.plot(mpc_t[:-1],lqr_u[:,1]/1000,'r',linestyle='dashed')
plt.plot(mpc_t[:-1],mpc_ol_u[:,1]/1000,'g',linestyle='dashed')
plt.plot(mpc_t[:-1],mpc_lqr_u[:,1]/1000,'k',linestyle='dashed')
plt.ylabel(r"$T_y$ [kN]")

plt.subplot(3,1,3)
plt.grid(True)
plt.plot(mpc_t[:-1],ol_u[:,2]/1000,'b')
plt.plot(mpc_t[:-1],lqr_u[:,2]/1000,'r',linestyle='dashed')
plt.plot(mpc_t[:-1],mpc_ol_u[:,2]/1000,'g',linestyle='dashed')
plt.plot(mpc_t[:-1],mpc_lqr_u[:,2]/1000,'k',linestyle='dashed')
plt.xlabel("t [s]")
plt.ylabel(r"$T_z$ [kN]")
plt.savefig(os.path.join(figure_dir,"3dof_control.png"))