import numpy as np
import matplotlib.pyplot as plt

# x_6dof = np.load("output/trajectory/001/X.npy")
x_6dof = np.load("data/x_6dof.npy")
# x_6dof = x_6dof[1:4,:]
x_3dofAttitude = np.load("data/pos.npy")


plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_3dofAttitude[:,0],x_3dofAttitude[:,1],x_3dofAttitude[:,2])
ax.plot3D(x_6dof[1,:],x_6dof[2,:],x_6dof[3,:])
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_aspect('equal')
plt.legend(["3DOF", "6DOF"])
plt.savefig("figures/compare_trajectory.png")
plt.show()