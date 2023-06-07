import os
import argparse

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import numpy as np

import control
from scipy.integrate import odeint

parser = argparse.ArgumentParser(description = "OL or LQR?")
parser.add_argument('-open_loop', action='store_true')

argument = parser.parse_args()

## Data & Figures
closed_loop = not argument.open_loop

file_dir = os.path.dirname(os.path.abspath(__file__))
traj_dir = os.path.join(file_dir, '../../data/', '3dof_trajectory_pointing_constraints')

if closed_loop:
    data_dir = os.path.join(file_dir, '../../data/', '3dof_lqr')
    figure_dir = os.path.join(file_dir, '../../figures/', '3dof_lqr')
else:
    data_dir = os.path.join(file_dir, '../../data/', '3dof_ol')
    figure_dir = os.path.join(file_dir, '../../figures/', '3dof_ol')

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
    g = jnp.array([0.0,0.0,-9.81])
    α = 3.9246E-4
    vx, vy, vz = x[3:6]
    v = jnp.linalg.norm(x[3:6])
    B = 2.5 / x[-1]
    return jnp.array([
        x[3],
        x[4],
        x[5],
        g[0] + u[0]/x[6] - B*v*vx,
        g[1] + u[1]/x[6] - B*v*vy,
        g[2] + u[2]/x[6] - B*v*vz,
        -α*jnp.linalg.norm(u)
    ])

# Load Trajectory Data
mass = np.array([np.load(os.path.join(traj_dir, "mass.npy"))]).T
pos = np.load(os.path.join(traj_dir, "pos.npy"))
vel = np.load(os.path.join(traj_dir, "vel.npy"))
# x_ol = np.load(os.path.join(traj_dir, "stateErrorOL_drag.npy"))
x_data = np.hstack((pos,vel,mass))

u_data = np.load(os.path.join(traj_dir, "thrust.npy"))

## LQR Settings
n = 7
m = 3
Q = 10000*np.identity(n)
Q[2,2] = 10000
R = 0.01*np.identity(m)

tf = 48
dt = 1
N = int(np.ceil(tf/dt) + 1)
ts = np.arange(0,tf+dt,dt)

f = jax.jit(dynamics)
fd = jax.jit(lambda x,u,dt=dt: x + dt*f(x,u))

# Generate K matrices using LQE
K = np.zeros((N,m,n))
A = np.zeros((N,n,n))
B = np.zeros((N,n,m))
for i in range(N):
    A[i],B[i] = linearize(fd,x_data[i],u_data[i])
    K[i] = control.dlqr(A[i],B[i],Q,R)[0]

# Simulate
f_sim = lambda x,t,u: dynamics(x,u)

x       = np.zeros((N,n))
e       = np.zeros((N,n))
u_ol    = np.zeros((N-1,m))
u_cl    = np.zeros((N-1,m))
u       = np.zeros((N-1,m))
x[0] = x_data[0]
for i in range(N-1):
    e[i]    = x[i] - x_data[i]
    u_ol[i] = u_data[i]
    if closed_loop:
        u_cl[i] = -K[i]@e[i]
    u[i]    = u_ol[i] + u_cl[i]
    x[i+1] = odeint(f_sim,x[i],[ts[i],ts[i+1]],args=(u[i],))[-1]

## Save results
np.save(os.path.join(data_dir, "x.npy"),x)
np.save(os.path.join(data_dir, "u.npy"),u_ol+u_cl)

## Plot figures
t_span = np.linspace(0,tf,N)
plt.figure()
plt.subplot(3,1,1)
plt.grid(True)
plt.plot(t_span,x_data[:,0])
plt.plot(t_span,x[:,0])
plt.ylabel(r"$r_x$ [m]")
plt.subplot(3,1,2)
plt.grid(True)
plt.plot(t_span,x_data[:,1])
plt.plot(t_span,x[:,1])
plt.ylabel(r"$r_y$ [m]")
plt.subplot(3,1,3)
plt.grid(True)
plt.plot(t_span,x_data[:,2])
plt.plot(t_span,x[:,2])
plt.legend(["Nominal","True"])
plt.xlabel("t [s]")
plt.ylabel(r"$r_z$ [m]")
plt.savefig(os.path.join(figure_dir,"3dof_pos_lqr.png"))

plt.figure()
plt.subplot(3,1,1)
plt.grid(True)
plt.plot(t_span[:-1],u_ol[:,0]/1000,'b')
plt.plot(t_span[:-1],u_ol[:,0]/1000 + u_cl[:,0]/1000,'r',linestyle='dashed')
plt.ylabel(r"$T_x$ [kN]")

plt.subplot(3,1,2)
plt.grid(True)
plt.plot(t_span[:-1],u_ol[:,1]/1000,'b')
plt.plot(t_span[:-1],u_ol[:,1]/1000 + u_cl[:,1]/1000,'r',linestyle='dashed')
plt.ylabel(r"$T_y$ [kN]")

plt.subplot(3,1,3)
plt.grid(True)
plt.plot(t_span[:-1],u_ol[:,2]/1000,'b')
plt.plot(t_span[:-1],u_ol[:,2]/1000 + u_cl[:,2]/1000,'r',linestyle='dashed')
plt.legend(["Nominal","True"])
plt.xlabel("t [s]")
plt.ylabel(r"$T_z$ [kN]")
plt.savefig(os.path.join(figure_dir,"3dof_control_lqr.png"))

