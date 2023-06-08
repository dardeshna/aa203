import os 

import numpy as np
import matplotlib.pyplot as plt

file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(file_dir, '../data')
figure_dir = os.path.join(file_dir, '../figures/', '3dof_vs_6dof_comparison')

sixdof_data_dir = os.path.join(data_dir, '6dof_trajectory')
traj_data_dir = os.path.join(data_dir, '3dof_trajectory_pointing_constraints')

x_6dof = np.load(os.path.join(sixdof_data_dir,"x_6dof.npy"))
x_3dofAttitude = np.load(os.path.join(traj_data_dir, "pos.npy"))


plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_3dofAttitude[:,0],x_3dofAttitude[:,1],x_3dofAttitude[:,2])
ax.plot3D(x_6dof[1,:],x_6dof[2,:],x_6dof[3,:])
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_aspect('equal')
plt.legend(["3DOF", "6DOF"])
plt.savefig(os.path.join(figure_dir,"compare_trajectory.png"))
plt.show()