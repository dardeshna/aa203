import numpy as np
from scipy.integrate import odeint


class FirstOrderHold:
    def __init__(self, m, K):
        self.K = K
        self.m = m
        self.n_x = m.n_x
        self.n_u = m.n_u

        self.A_bar = np.zeros([m.n_x * m.n_x, K - 1])
        self.B_bar = np.zeros([m.n_x * m.n_u, K - 1])
        self.C_bar = np.zeros([m.n_x * m.n_u, K - 1])
        self.S_bar = np.zeros([m.n_x, K - 1])
        self.z_bar = np.zeros([m.n_x, K - 1])

        # vector indices for flat matrices
        x_end = m.n_x
        A_bar_end = m.n_x * (1 + m.n_x)
        B_bar_end = m.n_x * (1 + m.n_x + m.n_u)
        C_bar_end = m.n_x * (1 + m.n_x + m.n_u + m.n_u)
        S_bar_end = m.n_x * (1 + m.n_x + m.n_u + m.n_u + 1)
        z_bar_end = m.n_x * (1 + m.n_x + m.n_u + m.n_u + 2)
        self.x_ind = slice(0, x_end)
        self.A_bar_ind = slice(x_end, A_bar_end)
        self.B_bar_ind = slice(A_bar_end, B_bar_end)
        self.C_bar_ind = slice(B_bar_end, C_bar_end)
        self.S_bar_ind = slice(C_bar_end, S_bar_end)
        self.z_bar_ind = slice(S_bar_end, z_bar_end)

        self.f, self.A, self.B = m.get_equations()

        # integration initial condition
        self.V0 = np.zeros((m.n_x * (1 + m.n_x + m.n_u + m.n_u + 2),))
        self.V0[self.A_bar_ind] = np.eye(m.n_x).reshape(-1)

        self.dt = 1. / (K - 1)

    def calculate_discretization(self, X, U, sigma):
        """
        Calculate discretization for given states, inputs and total time.

        :param X: Matrix of states for all time points
        :param U: Matrix of inputs for all time points
        :param sigma: Total time
        :return: The discretization matrices
        """
        for k in range(self.K - 1):
            self.V0[self.x_ind] = X[:, k]
            V = np.array(odeint(self._ode_dVdt, self.V0, (0, self.dt), args=(U[:, k], U[:, k + 1], sigma))[1, :])

            # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
            # flatten matrices in column-major (Fortran) order for CVXPY
            Phi_k1_k = V[self.A_bar_ind].reshape((self.n_x, self.n_x))
            self.A_bar[:, k] = Phi_k1_k.flatten(order='F')
            self.B_bar[:, k] = (Phi_k1_k @ V[self.B_bar_ind].reshape((self.n_x, self.n_u))).flatten(order='F')
            self.C_bar[:, k] = (Phi_k1_k @ V[self.C_bar_ind].reshape((self.n_x, self.n_u))).flatten(order='F')
            self.S_bar[:, k] = (Phi_k1_k @ V[self.S_bar_ind])
            self.z_bar[:, k] = (Phi_k1_k @ V[self.z_bar_ind])

        return self.A_bar, self.B_bar, self.C_bar, self.S_bar, self.z_bar

    def _ode_dVdt(self, V, t, u_t0, u_t1, sigma):
        """
        ODE function to compute dVdt.

        :param V: Evaluation state V = [x, Phi_A, B_bar, C_bar, S_bar, z_bar]
        :param t: Evaluation time
        :param u_t0: Input at start of interval
        :param u_t1: Input at end of interval
        :param sigma: Total time
        :return: Derivative at current time and state dVdt
        """
        alpha = (self.dt - t) / self.dt
        beta = t / self.dt
        x = V[self.x_ind]
        u = u_t0 + (t / self.dt) * (u_t1 - u_t0)

        # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
        # and pre-multiplying with \Phi_A(\tau_{k+1},\tau_k) after integration
        Phi_xi_k = V[self.A_bar_ind].reshape((self.n_x, self.n_x))

        A_subs = sigma * self.A(x, u)
        B_subs = sigma * self.B(x, u)
        f_subs = self.f(x, u).ravel()

        z_t = - A_subs @ x - B_subs @ u

        # use np.linalg.solve instead of np.linalg.inv to compute \Phi_A(\xi,\tau_k)^{-1} @ (...)
        res = np.linalg.solve(Phi_xi_k, np.column_stack((B_subs, f_subs, z_t)))
        Phi_k_xi_B = res[:, :-2]
        Phi_k_xi_f = res[:, -2]
        Phi_k_xi_z = res[:, -1]

        dVdt = np.zeros_like(V)
        dVdt[self.x_ind] = sigma * f_subs
        dVdt[self.A_bar_ind] = (A_subs @ Phi_xi_k).ravel()
        dVdt[self.B_bar_ind] = Phi_k_xi_B.ravel() * alpha
        dVdt[self.C_bar_ind] = Phi_k_xi_B.ravel() * beta
        dVdt[self.S_bar_ind] = Phi_k_xi_f
        dVdt[self.z_bar_ind] = Phi_k_xi_z

        return dVdt

    def integrate_nonlinear_piecewise(self, X_l, U, sigma):
        """
        Piecewise integration to verfify accuracy of linearization.
        :param X_l: Linear state evolution
        :param U: Linear input evolution
        :param sigma: Total time
        :return: The piecewise integrated dynamics
        """
        X_nl = np.zeros_like(X_l)
        X_nl[:, 0] = X_l[:, 0]

        for k in range(self.K - 1):
            X_nl[:, k + 1] = odeint(self._dx, X_l[:, k],
                                    (0, self.dt * sigma),
                                    args=(U[:, k], U[:, k + 1], sigma))[1, :]

        return X_nl

    def integrate_nonlinear_full(self, x0, U, sigma):
        """
        Simulate nonlinear behavior given an initial state and an input over time.
        :param x0: Initial state
        :param U: Linear input evolution
        :param sigma: Total time
        :return: The full integrated dynamics
        """
        X_nl = np.zeros([x0.size, self.K])
        X_nl[:, 0] = x0

        for k in range(self.K - 1):
            X_nl[:, k + 1] = odeint(self._dx, X_nl[:, k],
                                    (0, self.dt * sigma),
                                    args=(U[:, k], U[:, k + 1], sigma))[1, :]

        return X_nl

    def _dx(self, x, t, u_t0, u_t1, sigma):
        u = u_t0 + (t / (self.dt * sigma)) * (u_t1 - u_t0)

        return np.squeeze(self.f(x, u))
