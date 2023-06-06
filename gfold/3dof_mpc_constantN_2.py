import jax
import jax.numpy as jnp

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

import control
from scipy.integrate import odeint
from tqdm import tqdm

def linearize(f, x, u):
    """Linearize the function `f(s, u)` around `(s, u)`.

    Arguments
    ---------
    f : callable
        A nonlinear function with call signature `f(s, u)`.
    s : numpy.ndarray
        The state (1-D).
    u : numpy.ndarray
        The control input (1-D).

    Returns
    -------
    A : numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `s`.
    B : numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `u`.
    """
    # WRITE YOUR CODE BELOW ###################################################
    # INSTRUCTIONS: Use JAX to compute `A` and `B` in one line.
    A,B = jax.jacobian(f,[0,1])(x,u)

    return A, B

def dynamics(x, u):
    """
    x = numpy.1darray, r = x[0:3], v = x[3:6], m = x[6]
    g = [gx, gy, gz]
    """
    g = jnp.array([0,0,-9.80665])
    α = 5E-4
    return jnp.array([
        x[3],
        x[4],
        x[5],
        g[0] + u[0]/x[6],
        g[1] + u[1]/x[6],
        g[2] + u[2]/x[6],
        -α*jnp.linalg.norm(u)
    ])

def run_mpc(  N,                # receding horizon (parameter)
              α, ρ1, ρ2,        # Constraint parameters
              r0, v0, wet_mass, # initial constraints (parameters)
              rf, vf,           # terminal contraints
              dt, g, γ, θ, n, m_dry    # constants
    ):
    # CVX variables
    r = cp.Variable((N,3))
    v = cp.Variable((N,3))
    u = cp.Variable((N,3))
    z = cp.Variable(N)
    σ = cp.Variable(N)
    
    ## Objective
    objective = 0
    objective += -z[N-1]
    objective += 1000*cp.norm2(rf - r[N-1])
    objective += 1000*cp.norm2(vf - v[N-1])
    
    # Try to get to goal at each time step
    for i in range(N):
        objective += 100*cp.norm2(rf - r[i])
        objective += 100*cp.norm2(vf - v[i])
    
    constraints = []
    
    ## Terminal Constraints
    constraints += [
        # r[N-1] == rf,
        # v[N-1] == vf,
        z[N-1] >= np.log(m_dry)
    ]
    
    ## Initial Condition Constraints
    constraints += [
        r[0] == r0,
        v[0] == v0,
        z[0] == np.log(wet_mass)
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
    objective = cp.Minimize(objective)
    prob = cp.Problem(objective,constraints)
    prob.solve()
    status = prob.status
    
    if status == 'infeasible':
        return 0,0,0,0, status

    ## Acceleration to thrust
    m = np.exp(z.value)
    Tx = u.value[:,0] * m
    Ty = u.value[:,1] * m
    Tz = u.value[:,2] * m
    T = np.array([Tx,Ty,Tz]).T
    
    return r.value, v.value, m, T, status

# # New Shepard
# g0 = 9.80665
# dry_mass = 20569
# wet_mass = 25000 # Assumed from start of landing burn
# Isp = 260
# max_throttle = 0.8 # Safety
# min_throttle = 0.1
# g = np.array([0.0,0.0,-9.81])
# T_max = 490000
# φ = 0*np.deg2rad(1)
# γ = np.deg2rad(5)
# n = np.array([0,0,1])
# θ = np.deg2rad(20)

# ## Initial Conditions
# r0 = 1000 * np.array([0.5, 0.1, 2])
# v0 = np.array([20,0.01,-75])
# ## Terminal Conditions set to origin
# rf = np.array([0,0,0])
# vf = np.array([0,0,0])
# ## Time of Flight
# tf = 30
# dt = 1

g0 = 9.80665
dry_mass = 20569
wet_mass = 30000 # Assumed from start of landing burn
Isp = 260
max_throttle = 0.8 # Safety
min_throttle = 0.1
g = np.array([0.0,0.0,-9.81])
T_max = 490000
φ = 0*np.deg2rad(1)
γ = np.deg2rad(20)
n = np.array([0,0,1])
θ = np.deg2rad(20)

## Initial Conditions
r0 = 1000 * np.array([1.5, 0.5, 2])
v0 = np.array([50,-30,-100])
## Terminal Conditions set to origin
rf = np.array([0,0,0])
vf = np.array([0,0,0])
## Time of Flight
tf = 48
dt = 1
# N = 200
# dt = tf/N

N0 = int(np.ceil(tf/dt) + 1)
ts = np.arange(0,tf+dt,dt)

## Constraint Parameters
α = 5E-4
ρ1 = min_throttle*T_max
ρ2 = max_throttle*T_max

N = N0

N_fixed = 20

## Looping vars
s = 7
m = 3
x_hist  = np.zeros((N0,s))
u_hist  = np.zeros((N0-1,m))

x_hist[0] = np.concatenate((r0, v0, [wet_mass]))

## Last time MPC was feasible
t_prev = 0
x_prev = np.array([])
u_prev = np.array([])

## Iteratively solve Convex Problem
f_sim = lambda x,t,u: dynamics(x,u)
objective_choice = 'fuel_opt'

for t in tqdm(range(N0-1)):
    # Solve Convex Problem
    r, v, mass, u, status = run_mpc(N_fixed, α, ρ1, ρ2, r0, v0, wet_mass,rf, vf, dt, g, γ, θ, n, dry_mass)
    
    if status == 'infeasible':
        break
    else:
        t_prev = t
        x_prev = np.concatenate((r,v,mass[:, None]), axis=1)
        u_prev = u
        
        u_ctrl = u[0]
    
    # Update
    x_hist[t+1] = odeint(f_sim,x_hist[t],[0, dt], args=(u_ctrl,))[-1]
    u_hist[t]   = u_ctrl      

    N = int(N - dt)
    r0 = x_hist[t+1, 0:3]
    v0 = x_hist[t+1, 3:6]
    wet_mass = x_hist[t+1,6]

print(r0)
print(v0)
print(wet_mass)

## Use LQR to track remaining trajectory
print("Finish up what's left with LQR tracking")
Q = 10000*np.identity(s)
Q[2,2] = 100000
R = 0.01*np.identity(m)
closed_loop = True

f = jax.jit(dynamics)
fd = jax.jit(lambda x,u,dt=dt: x + dt*f(x,u))

# Generate K matrices using LQE
N = N + 1

K = np.zeros((N,m,s))
A = np.zeros((N,s,s))
B = np.zeros((N,s,m))
for i in range(N):
    A[i],B[i] = linearize(fd,x_prev[i],u_prev[i])
    K[i] = control.dlqr(A[i],B[i],Q,R)[0]

# Simulate remaining trajectory
x       = np.zeros((N,s))
e       = np.zeros((N,s))
u_ol    = np.zeros((N-1,m))
u_cl    = np.zeros((N-1,m))
u       = np.zeros((N-1,m))

x[0]    = x_prev[0]

for i in range(N-1):
    e[i]    = x[i] - x_prev[i]
    u_ol[i] = u_prev[i]
    if closed_loop:
        u_cl[i] = -K[i]@e[i]
    u[i]    = u_ol[i] + u_cl[i]
    x[i+1] = odeint(f_sim,x[i],[0,dt],args=(u[i],))[-1]
    
## Merge with MPC
x_hist[t_prev:] = x
u_hist[t_prev:] = u

print(x_hist[-1])

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