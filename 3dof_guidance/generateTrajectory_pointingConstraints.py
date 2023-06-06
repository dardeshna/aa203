import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

## Constants
# Mars
# dry_mass = 1700
# Isp = 225
# max_throttle = 0.8
# min_throttle = 0.2
# g = np.array([0,0,-3.7114]) # Mars Gravity
# T_max = 24000
# γ = 4*np.deg2rad(1) # glide slope constraint
# wet_mass = 2000
# n = np.array([0,0,1]) # Nominal thrust pointing vector
# θ = np.deg2rad(45) # Attitude constraint
## Initial Conditions
# r0 = np.array([450, -330, 2400])
# v0 = np.array([-40,10,-10])
# ## Terminal Conditions set to origin
# rf = np.array([0,0,0])
# vf = np.array([0,0,0])
# ## Time of Flight
# tf = 57
# dt = 1

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
dt = 0.1
# N = 200
# dt = tf/N


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


np.save("data/pos.npy",r.value)
np.save("data/vel.npy",v.value)
np.save("data/mass.npy",m)

print(m[-1])

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(r.value[:,0],r.value[:,1],r.value[:,2])
ax.set_aspect('equal')
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
plt.savefig("figures/3dof_trajectory_attitude.png")

plt.figure()
plt.plot(r.value[:,0],r.value[:,1])
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.savefig("figures/3dof_surfacetrajectory_attitude.png")

t_span = np.linspace(0,tf,N)
plt.figure()
plt.subplot(3,1,1)
plt.plot(t_span,r.value[:,0])
plt.ylabel("rx [m]")
plt.subplot(3,1,2)
plt.plot(t_span,r.value[:,1])
plt.ylabel("ry [m]")
plt.subplot(3,1,3)
plt.plot(t_span,r.value[:,2])
plt.xlabel("t [s]")
plt.ylabel("rz [m]")
plt.savefig("figures/3dof_pos_attitude.png")

plt.figure()
plt.subplot(3,1,1)
plt.plot(t_span,v.value[:,0])
plt.ylabel("vx [m/s]")
plt.subplot(3,1,2)
plt.plot(t_span,v.value[:,1])
plt.ylabel("vy [m/s]")
plt.subplot(3,1,3)
plt.plot(t_span,v.value[:,2])
plt.xlabel("t [s]")
plt.ylabel("vz [m/s]")
plt.savefig("figures/3dof_vel_attitude.png")


Tx = u.value[:,0] * m * r_scale
Ty = u.value[:,1] * m * r_scale
Tz = u.value[:,2] * m * r_scale
T = np.array([Tx,Ty,Tz]).T
np.save("data/thrust.npy",T)
T = np.linalg.norm(T,2,1)

print(Tx[-1])
print(Ty[-1])
print(Tz[-1])


# plt.figure()
# plt.subplot(5,1,1)
# plt.plot(t_span,Tx)
# plt.ylabel("Tx [N]")
# plt.subplot(5,1,2)
# plt.plot(t_span,Ty)
# plt.ylabel("Ty [N]")
# plt.subplot(5,1,3)
# plt.plot(t_span,Tz)
# plt.ylabel("Tz [N]")
# plt.subplot(5,1,4)
# plt.plot(t_span,T)
# plt.ylabel("T [N]")
# plt.subplot(5,1,5)
# plt.plot(t_span,m)
# plt.ylabel("m [kg]")
# plt.xlabel("t [s]")

plt.figure()
plt.subplot(2,1,1)
plt.plot(t_span,T)
plt.plot(t_span,ρ1*np.array([1]*len(t_span)))
plt.plot(t_span,ρ2*np.array([1]*len(t_span)))
plt.legend(["T","T_max","T_min"])
plt.ylabel("T [N]")
plt.subplot(2,1,2)
plt.plot(t_span,m)
plt.ticklabel_format(style='plain')
plt.ylabel("m [kg]")
plt.xlabel("t [s]")
plt.savefig("figures/3dof_control_attitude.png")

plt.figure()
plt.subplot(3,1,1)
plt.plot(t_span,Tx)
plt.ylabel("Tx [N]")
plt.subplot(3,1,2)
plt.plot(t_span,Ty)
plt.ylabel("Ty [N]")
plt.subplot(3,1,3)
plt.plot(t_span,Tz)
plt.xlabel("t [s]")
plt.ylabel("Tz [N]")
plt.savefig("figures/3dof_controlVector_attitude.png")

# plt.show()