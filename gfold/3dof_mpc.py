import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm

def dynamics(x, u):
    """
    x = numpy.1darray, r = x[0:3], v = x[3:6], m = x[6]
    g = [gx, gy, gz]
    """
    g = np.array([0,0,-3.7114])
    α = 5E-4
    return np.array([
        x[3],
        x[4],
        x[5],
        g[0] + u[0]/x[6],
        g[1] + u[1]/x[6],
        g[2] + u[2]/x[6],
        -α*np.linalg.norm(u)
    ])

def run_mpc(  N,                # receding horizon (parameter)
              α, ρ1, ρ2,        # Constraint parameters
              r0, v0, wet_mass, # initial constraints (parameters)
              rf, vf,           # terminal contraints
              dt, g, γ, θ, n,    # constants
              objective_choice
    ):
    # CVX variables
    r = cp.Variable((N,3))
    v = cp.Variable((N,3))
    u = cp.Variable((N,3))
    z = cp.Variable(N)
    σ = cp.Variable(N)
    
    constraints = []
    
    ## Objective
    if objective_choice == 'fuel_opt':
        objective = cp.Minimize(-z[N-1])
        ## Terminal Constraints
        constraints += [
            r[N-1] == rf,
            v[N-1] == vf
        ]
    elif objective_choice == 'error_opt':
        # objective = cp.Minimize(cp.norm2(r[N-1,0:2] - rf[0:2]))
        objective = cp.Minimize(cp.norm2(r[N-1] - rf))

        ## Terminal Constraints
        constraints += [
            # r[N-1,2] == rf[2],
            # v[N-1] == vf
        ]
    else:
        raise ValueError('Objective choice not recognized')
    
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
            # r[i][2] >= -0.001
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
        return 0, 0, 0, status
        objective_choice = 'error_opt'
        return run_mpc(N, α, ρ1, ρ2, r0, v0, wet_mass,rf, vf, dt, g, γ, θ, n,
                          objective_choice=objective_choice)
    
    m = np.exp(z.value)
    Tx = u.value[:,0] * m
    Ty = u.value[:,1] * m
    Tz = u.value[:,2] * m
    T = np.array([Tx,Ty,Tz]).T
    
    return r.value, v.value, T, status

## Constants
g0 = 9.80665
dry_mass = 1700
Isp = 225
max_throttle = 0.8
min_throttle = 0.2
g = np.array([0,0,-3.7114]) # Mars Gravity
T_max = 24000
γ = 4*np.deg2rad(1) # glide slope constraint
wet_mass = 2000
n = np.array([0,0,1]) # Nominal thrust pointing vector
θ = np.deg2rad(45) # Attitude constraint

## Initial Conditions
r0 = np.array([450, -330, 2400])
v0 = np.array([-40,10,-10])
## Terminal Conditions set to origin
rf = np.array([0,0,0])
vf = np.array([0,0,0])
## Time of Flight
tf = 57
dt_cvx = 0.1
dt_mpc = 1

N0_mpc = int(np.ceil(tf/dt_mpc) + 1)
ts = np.arange(0,tf,dt_mpc)

N0_cvx = int(np.ceil(tf/dt_cvx) + 1)
ts_cvx = np.arange(0, tf, dt_cvx)

## Constraint Parameters
α = 5E-4
ρ1 = min_throttle*T_max
ρ2 = max_throttle*T_max

N = N0_mpc

## Looping vars
s = 7
m = 3
x_hist  = np.zeros((N0_cvx,s))
u_hist  = np.zeros((N0_cvx-1,m))

x_hist = np.zeros((1,s))
u_hist = np.zeros((1,m))

x_hist[0] = np.concatenate((r0, v0, [wet_mass]))

# Iteratively solve Convex Problem
f_sim = lambda x,t,u: dynamics(x,u)
objective_choice = 'fuel_opt'

for t in tqdm(range(N0_mpc)):
    # Solve Convex Problem
    try:
        _, _, u, status = run_mpc(N0_cvx, α, ρ1, ρ2, r0, v0, wet_mass,rf, vf, dt_cvx, g, γ, θ, n,
                            objective_choice=objective_choice)
        
        if status == 'infeasible' and objective_choice == 'fuel_opt':
            raise Exception('Raised when ful opt is infeasible: ', N)
            objective_choice = 'error_opt'
            _, _, u, status = run_mpc(N, α, ρ1, ρ2, r0, v0, wet_mass,rf, vf, dt, g, γ, θ, n,
                            objective_choice=objective_choice)
            
            if status == 'infeasible':
                raise Exception('Raised when switching from fuel optimal to error optimal: ', N)
        elif status == 'infeasible' and objective_choice == 'error_opt':
            raise Exception('Raised when trying error optimal: ', N)
        
    except Exception as e:
        print(e)
        # x_hist = x_hist[0:t]
        # u_hist = u_hist[0:t-1]
        ts_cvx   = ts_cvx[0:len(x_hist)]
        break
    
    # Update
    for dt in range(int(dt_mpc / dt_cvx)):
        x_hist = np.append(x_hist,
                           [odeint(f_sim,x_hist[-1],[0, dt_cvx], args=(u[dt],))[-1]],
                           axis=0
                           )
        # x_hist[t+1] =  odeint(f_sim,x_hist[t],[ts[t], ts[t+1]], args=(u[0],))[-1]
        if t ==0 and dt == 0:
            u_hist = np.array([u[dt]])
        else:
            u_hist = np.append(u_hist, [u[dt]],axis = 0)
        

    N = int(N - dt_mpc)
    N0_cvx = int(N0_cvx - int(dt_mpc/dt_cvx))
    r0 = x_hist[-1, 0:3]
    v0 = x_hist[-1, 3:6]
    wet_mass = x_hist[-1,6]
   

print(r0)
print(v0)

## Plot
plt.figure()
plt.subplot(3,1,1)
plt.plot(ts_cvx,x_hist[:,0])
plt.ylabel("rx [m]")
plt.subplot(3,1,2)
plt.plot(ts_cvx,x_hist[:,1])
plt.ylabel("ry [m]")
plt.subplot(3,1,3)
plt.plot(ts_cvx,x_hist[:,2])
plt.xlabel("t [s]")
plt.ylabel("rz [m]")
plt.savefig("../figures/3dof_mpc_pos.png")

plt.figure()
plt.plot(x_hist[:,0],x_hist[:,1])
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.savefig("../figures/3dof_mpc_surfacetrajectory.png")

ts_cvx = ts_cvx[0:-1]
plt.figure()
plt.subplot(3,1,1)
plt.plot(ts_cvx,u_hist[:,0])
plt.ylabel("Tx [N]")
plt.subplot(3,1,2)
plt.plot(ts_cvx,u_hist[:,1])
plt.ylabel("Ty [N]")
plt.subplot(3,1,3)
plt.plot(ts_cvx,u_hist[:,2])
plt.xlabel("t [s]")
plt.ylabel("Tz [N]")
plt.savefig("../figures/3dof_mpc_control.png")