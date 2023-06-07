import jax
import jax.numpy as jnp

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from tqdm import tqdm

def dynamics(x, u, g, α):
    """
    x = numpy.1darray, r = x[0:3], v = x[3:6], m = x[6]
    g = [gx, gy, gz]
    """
    return jnp.array([
        x[3],
        x[4],
        x[5],
        g[0] + u[0]/x[6],
        g[1] + u[1]/x[6],
        g[2] + u[2]/x[6],
        -α*jnp.linalg.norm(u)
    ])

def run_mpc(r0, v0, wet_mass,           # Initial conditions
            rf, vf,                     # Final conditions
            dry_mass, g, θ, n, γ, dt,   # constants
            α, ρ1, ρ2,                  # constraint parameters
            N                           # local horizon
            ):
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
    objective = 0
    objective += -z[N-1]
    objective += cp.norm2(r[N-1] - rf)
    objective += cp.norm2(v[N-1] - vf)

    for i in range(N):
        objective += cp.norm2(r[i] - rf)
        objective += cp.norm2(v[i] - vf)

        objective += 10*cp.norm2(r[i,2] - rf[2])

    objective = cp.Minimize(objective)

    # objective = cp.Minimize(-z[N-1])

    constraints = []
    ## Initial Condition Constraints
    constraints += [
        r[0] == r0,
        v[0] == v0,
        z[0] == np.log(wet_mass)
    ]

    ## Terminal Constraints
    constraints += [
        # r[N-1] == rf,
        # v[N-1] == vf,
        z[N-1] >= np.log(dry_mass)
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
    status = prob.status

    if status == 'infeasible':
        return 0,0,0,0, status

    # Restoration
    # m = np.exp(z.value)
    m = m_scale*np.exp(z.value)
    r = r*r_scale
    v = v*r_scale

    # Acceleration to thrust
    Tx = u.value[:,0] * m * r_scale
    Ty = u.value[:,1] * m * r_scale
    Tz = u.value[:,2] * m * r_scale
    T = np.array([Tx,Ty,Tz]).T

    return r.value, v.value, m, T, status

def run_min_landing_mpc(r0, v0, wet_mass,           # Initial conditions
                        rf, vf,                     # Final conditions
                        dry_mass, g, θ, n, γ, dt,   # constants
                        α, ρ1, ρ2,                  # constraint parameters
                        N                           # local horizon
                        ):
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

    y = cp.Variable((N,7))

    ## Constants
    E   = np.concatenate((np.eye(3), np.zeros((3,4))), axis=1)
    F   = np.concatenate((np.zeros((1,6)), [1]), axis=1)
    E_u = np.concatenate((np.eye(3), np.zeros((3,1))), axis=1)
    E_v = np.concatenate((np.zeros((3,3)), np.eye(3), np.zeros((3,1))), axis=1)

    ## Objective
    objective = 0
    objective += cp.norm2(E @ y[N-1])

    objective = cp.Minimize(objective)

    ## Constraints
    constraints = []

    constraints += [
        F @ y[N-1] >= np.log(dry_mass)
    ]

    # Gamma = || Thrust ||




## New Shepard
dry_mass = 20569
wet_mass = 27000 # Assumed from start of landing burn
Isp = 260
max_throttle = 0.8 # Safety
min_throttle = 0.1
T_max = 490000
φ = 0*np.deg2rad(1)
γ = np.deg2rad(20)
n = np.array([0,0,1])
θ = np.deg2rad(27)

## Earth
g0 = 9.80665
g = np.array([0.0,0.0,-g0])

## Initial Conditions
r0 = 1000 * np.array([1.5, 0.5, 2])
v0 = np.array([50,-30,-100])
## Terminal Conditions set to origin
rf = np.array([0,0,0])
vf = np.array([0,0,0])
## Time of Flight
tf = 48
dt = 1

T = int(np.ceil(tf/dt) + 1)
ts = np.arange(0,tf+dt,dt)

N = 20
# N = T

## Constraint Parameters
α = 1/(g0*Isp*np.cos(φ))
ρ1 = min_throttle*T_max*np.cos(φ)
ρ2 = max_throttle*T_max*np.cos(φ)

## Looping vars
s = 7
m = 3
x_hist  = np.zeros((T,s))
u_hist  = np.zeros((T-1,m))

x_hist[0] = np.concatenate((r0, v0, [wet_mass]))

## Iteratively solve Convex Problem
f_sim = lambda x,t,u: dynamics(x,u,g,α)

for t in tqdm(range(T-1)):
    # Solve Convex Problem
    r, v, mass, u, status = run_mpc(r0, v0, wet_mass, rf, vf, dry_mass, g, θ, n, γ, dt, α, ρ1, ρ2, N)
    
    if status == 'infeasible':
        print(status)
        break
    else:
        # t_prev = t
        # x_prev = np.concatenate((r,v,mass[:, None]), axis=1)
        # u_prev = u
        
        u_ctrl = u[0]
    # print(r[1])
    # print(r[-1])
    # Update
    x_hist[t+1] = odeint(f_sim,x_hist[t],[0, dt], args=(u_ctrl,))[-1]
    u_hist[t]   = u_ctrl      

    # N = int(N - dt)
    r0 = x_hist[t+1, 0:3]
    v0 = x_hist[t+1, 3:6]
    wet_mass = x_hist[t+1,6]

print(r0)
print(v0)
print(wet_mass)

# print(r[-1])

## Plot
plt.figure()
plt.subplot(3,1,1)
plt.plot(ts,x_hist[:,0])
plt.ylabel("rx [m]")
plt.subplot(3,1,2)
plt.plot(ts,x_hist[:,1])
plt.ylabel("ry [m]")
plt.subplot(3,1,3)
plt.plot(ts,x_hist[:,2])
plt.xlabel("t [s]")
plt.ylabel("rz [m]")
plt.savefig("../figures/3dof_mpc_finish_lqr_pos.png")

plt.figure()
plt.plot(x_hist[:,0],x_hist[:,1])
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.savefig("../figures/3dof_mpc_finish_lqr_surfacetrajectory.png")

plt.figure()
plt.subplot(3,1,1)
plt.plot(ts,x_hist[:,3])
plt.ylabel("vx [m/s]")
plt.subplot(3,1,2)
plt.plot(ts,x_hist[:,4])
plt.ylabel("vy [m/s]")
plt.subplot(3,1,3)
plt.plot(ts,x_hist[:,5])
plt.xlabel("t [s]")
plt.ylabel("vz [m/s]")

ts = ts[0:-1]
plt.figure()
plt.subplot(3,1,1)
plt.plot(ts,u_hist[:,0])
plt.ylabel("Tx [N]")
plt.subplot(3,1,2)
plt.plot(ts,u_hist[:,1])
plt.ylabel("Ty [N]")
plt.subplot(3,1,3)
plt.plot(ts,u_hist[:,2])
plt.xlabel("t [s]")
plt.ylabel("Tz [N]")
plt.savefig("../figures/3dof_mpc_finish_lqr_control.png")

plt.show()