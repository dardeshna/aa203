from functools import partial

import jax
import jax.numpy as jnp

import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from tqdm.auto import tqdm

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

def dynamics(x, u, g, α, dt):
    """
    x = numpy.1darray, r = x[0:3], v = x[3:6], z = x[6], sigma = z[7]
    g = [gx, gy, gz]
    """

    dx_dt = jnp.array([
        x[3],                  # pos
        x[4],
        x[5],
        g[0] + u[0],           # vel
        g[1] + u[1],
        g[2] + u[2],
        -α*jnp.linalg.norm(u), # z = log(wet mass)
        0                      # sigma
    ])

    x_next = x + dt*dx_dt

    return x_next

@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0, 0))
def affinize(f, s, u):
    """Affinize the function `f(s, u)` around `(s, u)`."""
    # PART (b) ################################################################
    # INSTRUCTIONS: Use JAX to affinize `f` around `(s,u)` in two lines.

    A, B = jax.jacobian(f, (0,1))(s,u)
    c = f(s,u) - A@s - B@u

    # END PART (b) ############################################################
    return A, B, c

def scp_iteration(f,                                # dynamics
                  s0, s_goal,                       # initial and goal states
                  s_prev, u_prev,                   # warm start
                  P, Q, R,                          # SCP constants
                  α, ρ1, ρ2,                        # Constraint parameters
                  dt,γ, θ, thrust_ang               # constants
                  ):
    """Solve a single SCP sub-problem for the rocket landing problem."""
    n = s_prev.shape[-1]    # state dimension
    m = u_prev.shape[-1]    # control dimension
    N = u_prev.shape[0]     # number of steps

    Af, Bf, cf = affinize(f, s_prev[:-1], u_prev)
    Af, Bf, cf = np.array(Af), np.array(Bf), np.array(cf)

    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))

    # Construct the convex SCP sub-problem.

    objective = 0
    constraints = []

    # Terminal cost
    objective += cvx.quad_form(s_cvx[N,0:6] - s_goal, P)

    # Initial conditon
    constraints.append(s_cvx[0,:] == s0)

    # Terminal constraint
    # Maybe not needed?
    
    # Iterative building of objective and constraints
    for k in range(0,N):
        # Cost
        objective += cvx.quad_form(s_cvx[k,0:6] - s_goal, Q)
        objective += cvx.quad_form(u_cvx[k,:], R)       
        
        # Dynamics constraint
        constraints.append(s_cvx[k+1,0:7] == Af[k,0:7,0:7]@s_cvx[k,0:7] + Bf[k,0:7,:]@u_cvx[k] + cf[k,0:7])

        # Thrust magnitude
        constraints.append(cvx.norm2(u_cvx[k]) <= s_cvx[k,7])

        # Thrust pointing
        constraints.append(thrust_ang@u_cvx[k] >= jnp.cos(θ)*s_cvx[k,7])

        # Enforce sigma & z
        wet_mass = jnp.exp(s0[6])
        z0_innerTerm = wet_mass - α*ρ2*k*dt
        zupper_comp  = wet_mass - α*ρ1*k*dt

        z0 = np.log(z0_innerTerm)
        zupper = np.log(zupper_comp)

        μ1 = ρ1/z0_innerTerm
        μ2 = ρ2/z0_innerTerm

        constraints.append(s_cvx[k,7] >= μ1 * (1 - (s_cvx[k,6] - z0) + 1/2*cvx.square(s_cvx[k,6]-z0)))
        constraints.append(s_cvx[k,7] <= μ2 * (1 - (s_cvx[k,6] - z0)))
        constraints.append(s_cvx[k,6] >= z0)
        constraints.append(s_cvx[k,6] <= zupper)

        # Glide slope
        constraints.append(np.tan(γ) * cvx.norm2(s_cvx[k,0:2]) <= s_cvx[k,2],)

        # Solve Problem
        prob = cvx.Problem(cvx.Minimize(objective), constraints)
        prob.solve()

        if prob.status != 'optimal':
            raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)
        
        s = s_cvx.value
        u = u_cvx.value
        J = prob.objective.value

        return s, u, J
    
    # Solve Problem
    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve()
    if prob.status != 'optimal':
        raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)
    s = s_cvx.value
    u = u_cvx.value
    J = prob.objective.value
    return s, u, J

def solve_rocket_landing_scp(f, s0, s_goal, N, P, Q, R, eps, α, ρ1, ρ2, dt,γ, θ, thrust_ang,
                                 max_iters, 
                                 s_init=None, u_init=None,
                                 convergence_error=False):
    """Solve the rocket landing problem via SCP."""
    n = s0.shape[0]  # state dimension
    m = R.shape[0]  # control dimension

    # Initialize trajectory
    if s_init is None or u_init is None:
        s = np.zeros((N + 1, n))
        u = np.zeros((N, m))
        s[0] = s0
        for k in range(N):
            s[k+1] = f(s[k], u[k])
    else:
        s = np.copy(s_init)
        u = np.copy(u_init)

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    J = np.zeros(max_iters + 1)
    J[0] = np.inf
    for i in range(max_iters):
        s, u, J[i + 1] = scp_iteration(f, s0, s_goal, s, u, P, Q, R,
                                       α, ρ1, ρ2, dt,γ, θ, thrust_ang)
        dJ = np.abs(J[i + 1] - J[i])
        if dJ < eps:
            converged = True
            break
    if not converged and convergence_error:
        raise RuntimeError('SCP did not converge!')
    return s, u

## New Shepard
dry_mass = 20569
wet_mass = 25000 # Assumed from start of landing burn
Isp = 260
max_throttle = 0.8 # Safety
min_throttle = 0.1

T_max = 490000
φ = 0*np.deg2rad(1)
γ = np.deg2rad(5)
thrust_ang = np.array([0,0,1])
θ = np.deg2rad(20)

## Earth
g0 = 9.80665
g = np.array([0.0,0.0,-g0])

## Initial Conditions
r0 = 1000 * np.array([0.5, 0.1, 2])
v0 = np.array([20,0.01,-75])

s0 = np.concatenate((r0, v0, [np.log(wet_mass), 0]))

## Terminal Conditions set to origin
rf = np.array([0,0,0])
vf = np.array([0,0,0])

s_goal = np.concatenate((rf, vf))

## Time of Flight
tf = 30
dt = 0.1

tot_time = int(np.ceil(tf/dt) + 1)
ts = np.arange(0,tf+dt,dt)

## Constraint Parameters
α = 5E-4
ρ1 = min_throttle*T_max
ρ2 = max_throttle*T_max

## SCP Constants
s_dim  = 8                                  # state dim
a_dim  = 3                                  # action dim

P = 1e2*np.eye(s_dim-2)                       # terminal state cost matrix
Q = 1e1*np.eye(s_dim-2)                       # state cost matrix
R = 1e-2*np.eye(a_dim)                      # control cost matrix
eps = 1e-3                              # SCP convergence tolerance

N = 50
N_scp = 10


## Solve MPC thru SCP
f = partial(dynamics, g=g, α=α, dt=dt)

s_mpc  = np.zeros((tot_time, N + 1, s_dim))
u_mpc  = np.zeros((tot_time, N, a_dim))
s      = np.copy(s0)
s_init = None
u_init = None

total_control_cost = 0.

for t in tqdm(range(tot_time)):
    # Solve MPC problem at time t
    s_mpc[t], u_mpc[t] = solve_rocket_landing_scp(f, s, s_goal, N, P, Q, R, eps, 
                                                  α, ρ1, ρ2, dt,γ, θ, thrust_ang,
                                                  max_iters=N_scp,
                                                  s_init=s_init, u_init=u_init,
                                                  convergence_error=False)
    
    # Push the state `s` forward in time with a closed-loop MPC input
    s = f(s, u_mpc[t,0])

    # Accumulate the actual control cost
    total_control_cost += u_mpc[t, 0].T @ R @ u_mpc[t, 0]

    # Use this solution to warm-start the next iteration
    u_init = np.concatenate([u_mpc[t, 1:], u_mpc[t, -1:]])
    s_init = np.concatenate([
        s_mpc[t, 1:],
        f(s_mpc[t, -1], u_mpc[t, -1]).reshape([1, -1])
        ])

print('Total control cost:', total_control_cost)