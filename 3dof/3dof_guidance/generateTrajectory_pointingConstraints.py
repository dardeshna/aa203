import os
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

## Data & Figures
file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(file_dir, '../../data/', '3dof_trajectory_pointing_constraints')
figure_dir = os.path.join(file_dir, '../../figures/', '3dof_trajectory_pointing_constraints')

# New Shepard
g0 = 9.80665
dry_mass = 20569
wet_mass = 27000 # Assumed from start of landing burn
Isp = 260
max_throttle = 0.8 # Safety
min_throttle = 0.1
g = np.array([0.0,0.0,-9.81])
T_max = 490000
φ = 0*np.deg2rad(1)
γ = np.deg2rad(20)
n = np.array([0,0,1])
θ = np.deg2rad(27)

## Initial Conditions
r0 = 1000 * np.array([1.5, 0.5, 2])
v0 = np.array([50,-30,-100])
## Terminal Conditions set to origin
rf = np.array([0,0,0])
vf = np.array([0,0,0])
## Time of Flight
tf = 48
dt = 1

N = int(np.ceil(tf/dt) + 1)
ts = np.arange(0,tf,dt)

## Constraint Parameters
α = 1/(g0*Isp*np.cos(φ))
ρ1 = min_throttle*T_max*np.cos(φ)
ρ2 = max_throttle*T_max*np.cos(φ)

# Normalization
r_scale = np.linalg.norm(r0)
m_scale = wet_mass
α = α*r_scale
g = g/r_scale
ρ1 = ρ1/r_scale/m_scale
ρ2 = ρ2/r_scale/m_scale
wet_mass = wet_mass/m_scale
dry_mass = dry_mass/m_scale
r0 = r0/r_scale
v0 = v0/r_scale

#### Set up CVXPY Problem
r = cp.Variable((N,3))
v = cp.Variable((N,3))
u = cp.Variable((N,3))
z = cp.Variable(N)
σ = cp.Variable(N)

## Objective
objective = cp.Minimize(-z[N-1])

constraints = []
## Initial Condition Constraints
constraints += [
    r[0] == r0,
    v[0] == v0,
    z[0] == np.log(wet_mass)
]
## Terminal Constraints
constraints += [
    r[N-1] == rf,
    v[N-1] == vf
]
## Constraints at each time step
for i in range(N):
    z0_innerTerm = wet_mass - α*ρ2*i*dt
    zupper_comp  = wet_mass - α*ρ1*i*dt

    z0 = np.log(z0_innerTerm)
    zupper = np.log(zupper_comp)

    μ1 = ρ1/z0_innerTerm
    μ2 = ρ2/z0_innerTerm
    constraints += [
        # Thrust Magnitude Constraint
        cp.norm2(u[i]) <= σ[i],
        # Thrust Pointing Constraint
        n@u[i] >= np.cos(θ)*σ[i],
        # Enforce Sigma
        σ[i] >= μ1 * (1 - (z[i] - z0) + 1/2*cp.square(z[i]-z0)),
        σ[i] <= μ2 * (1 - (z[i] - z0)),
        # Enforce z
        z[i] >= z0,
        z[i] <= zupper,
        # Glide Slope
        np.tan(γ) * cp.norm2(r[i][0:2]) <= r[i][2],
        # Enforce that feasible trajectories don't go under the surface
        r[i][2] >= -0.001
    ]

## Dynamics at Each Time Step, using trapezoidal integration
for i in range(N-1):
    constraints += [
        # Velocity Update
        v[i+1] == v[i] + dt/2*(u[i] + u[i+1]) + g*dt,
        # Position Update
        r[i+1] == r[i] + dt/2*(v[i] + v[i+1]) + (dt**2)/2*(u[i+1]-u[i]),
        # z Update
        z[i+1] == z[i] - α*dt/2*(σ[i] + σ[i+1])
    ]

## Solve Problem
prob = cp.Problem(objective,constraints)
prob.solve()
print(prob.status)

# Restoration
m = m_scale*np.exp(z.value)
r = r*r_scale
v = v*r_scale

Tx = u.value[:,0] * m * r_scale
Ty = u.value[:,1] * m * r_scale
Tz = u.value[:,2] * m * r_scale
T = np.array([Tx,Ty,Tz]).T

## Save data
np.save(os.path.join(data_dir, "pos.npy"),r.value)
np.save(os.path.join(data_dir, "vel.npy"),v.value)
np.save(os.path.join(data_dir, "mass.npy"),m)
np.save(os.path.join(data_dir, "thrust.npy"),T)

T = np.linalg.norm(T,2,1)

print(r.value[-1])
print(m[-1])
print(np.linalg.norm(r.value[-1]))
print(np.linalg.norm(v.value[-1]))
# print(Tx[-1])
# print(Ty[-1])
# print(Tz[-1])

## Generate plots
plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(r.value[:,0],r.value[:,1],r.value[:,2])
ax.set_aspect('equal')
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
plt.savefig(os.path.join(figure_dir, "3dof_trajectory_attitude.png"))

plt.figure()
plt.plot(r.value[:,0],r.value[:,1])
plt.grid(True)
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.savefig(os.path.join(figure_dir, "3dof_surfacetrajectory_attitude.png"))

t_span = np.linspace(0,tf,N)
plt.figure()
plt.subplot(3,1,1)
plt.grid(True)
plt.plot(t_span,r.value[:,0])
plt.ylabel(r"$r_x$ [m]")
plt.subplot(3,1,2)
plt.grid(True)
plt.plot(t_span,r.value[:,1])
plt.ylabel(r"$r_y$ [m]")
plt.subplot(3,1,3)
plt.grid(True)
plt.plot(t_span,r.value[:,2])
plt.xlabel("t [s]")
plt.ylabel(r"$r_z$ [m]")
plt.savefig(os.path.join(figure_dir, "3dof_pos_attitude.png"))

plt.figure()
plt.subplot(3,1,1)
plt.grid(True)
plt.plot(t_span,v.value[:,0])
plt.ylabel(r"$v_x$ [m/s]")
plt.subplot(3,1,2)
plt.grid(True)
plt.plot(t_span,v.value[:,1])
plt.ylabel(r"$v_y$ [m/s]")
plt.subplot(3,1,3)
plt.grid(True)
plt.plot(t_span,v.value[:,2])
plt.xlabel("t [s]")
plt.ylabel(r"$v_z$ [m/s]")
plt.savefig(os.path.join(figure_dir, "3dof_vel_attitude.png"))

plt.figure()
plt.subplot(2,1,1)
plt.grid(True)
plt.plot(t_span,T/1000)
plt.plot(t_span,ρ1*r_scale*m_scale*np.array([1]*len(t_span))/1000,'k',linestyle='--')
plt.plot(t_span,ρ2*r_scale*m_scale*np.array([1]*len(t_span))/1000,'k',linestyle='--')
plt.legend(["Thrust","Max Thrust","Min Thrust"])
plt.ylabel("Thrust [kN]")
plt.subplot(2,1,2)
plt.grid(True)
plt.plot(t_span,m)
plt.ticklabel_format(style='plain')
plt.ylabel("Mass [kg]")
plt.xlabel("t [s]")
plt.savefig(os.path.join(figure_dir, "3dof_control_attitude.png"))

plt.figure()
plt.subplot(3,1,1)
plt.grid(True)
plt.plot(t_span,Tx/1000)
plt.ylabel(r"$T_x$ [kN]")
plt.subplot(3,1,2)
plt.grid(True)
plt.plot(t_span,Ty/1000)
plt.ylabel(r"$T_y$ [kN]")
plt.subplot(3,1,3)
plt.grid(True)
plt.plot(t_span,Tz/1000)
plt.xlabel("t [s]")
plt.ylabel(r"$T_z$ [kN]")
plt.savefig(os.path.join(figure_dir, "3dof_controlVector_attitude.png"))

plt.show()