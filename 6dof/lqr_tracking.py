import matplotlib.pyplot as plt

import numpy as np
import scipy.interpolate

from scipy.integrate import odeint

from Models.rocket_landing_3d import Model
from Models.rocket_landing_3d_plot import plot

from global_parameters import K

m = Model()

# state and input
X = np.empty(shape=[m.n_x, K])
U = np.empty(shape=[m.n_u, K])

traj_folder = "001"
X = np.load(f"output/trajectory/{traj_folder}/X.npy")[-1]
U = np.load(f"output/trajectory/{traj_folder}/U.npy")[-1]
t_f = np.load(f"output/trajectory/{traj_folder}/sigma.npy")[-1]

X_interp = scipy.interpolate.interp1d(np.linspace(0, t_f, K), X)
U_interp = scipy.interpolate.interp1d(np.linspace(0, t_f, K), U)

Q = np.diag([0.1, 10000, 10000, 10000, 1, 1, 1, 100, 100, 100, 100, 1, 1, 1])
R = 0.01*np.identity(m.n_u+1)

t = 0
dt = 0.05
x = X[:, 0]

xs = []
us = []

f_func, A_func, B_func = m.get_equations()

while t < t_f:

    xs.append(x)

    x_ref = X_interp(t)
    u_ref = U_interp(t)

    A, B = A_func(x, u_ref), B_func(x, u_ref)

    # add fake input so roll is stabilizable
    B_roll = np.zeros(m.n_x)
    B_roll[13] = 1

    B = np.column_stack((B, B_roll))

    # continuous gains are close enough
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K_cl = np.linalg.solve(R, B.T @ P)

    print("time:", t)
    # print("err:", x_ref-x)
    # print("ref:", u_ref)
    # print("cl:", (K_cl @ (x_ref-x))[:-1])
    # print()

    # ignore fake input for roll rate
    u = u_ref + (K_cl @ (x_ref-x))[:-1]

    us.append(u)

    x = odeint(lambda y, t, u: f_func(y, u).ravel(), x, [t, t+dt], args=(u,))[-1]

    t += dt

xs = np.array(xs).T
us = np.array(us).T

# plot(xs[None,...], us[None,...], np.array([t_f])[None,...])
# plot(X[None,...], U[None,...], np.array([t_f])[None,...])

N = np.shape(xs)[1]

from matplotlib import pyplot as plt

for i in range(1,4):
    plt.figure()
    plt.plot(np.linspace(0, t_f, N), xs[i])
    plt.plot(np.linspace(0, t_f, K), X[i])
    plt.title(f"x[{i}]")

plt.show()