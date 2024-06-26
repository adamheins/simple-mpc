#!/usr/bin/env python3

# Simulation of comparison between PID and MPC for linear systems.

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


class LinearSystem:
    """Linear system of the form
    x_{k+1} = A*x_k + B*u_k
    y_k     = C*x_k"""

    def __init__(self, A, B):
        self.A = A
        self.B = B

        self.nx = A.shape[0]  # state dimension
        self.nu = B.shape[1]  # input dimension

    def step(self, x, u):
        u = np.minimum(u, 1.0)
        u = np.maximum(u, -1.0)
        return self.A @ x + self.B @ u

    def lookahead(self, x0, xs_des, Q, R):
        # TODO this is for single shooting
        # TODO this can be cleaned up a lot! (almost everything can be
        # pre-computed)
        N = xs_des.shape[0]

        Abar = np.zeros((self.nx * N, self.nx))
        Bbar = np.zeros((self.nx * N, self.nu * N))
        Qbar = np.zeros((self.nx * N, self.nx * N))
        Rbar = np.zeros((self.nu * N, self.nu * N))

        # TODO right now we don't include a separate QN for terminal condition
        # Construct matrices
        for k in range(N):
            l = k * self.nx
            u = (k + 1) * self.nx

            Abar[k * self.nx : (k + 1) * self.nx, :] = np.linalg.matrix_power(self.A, k + 1)
            Qbar[k * self.nx : (k + 1) * self.nx, k * self.nx : (k + 1) * self.nx] = Q
            Rbar[k * self.nu : (k + 1) * self.nu, k * self.nu : (k + 1) * self.nu] = R

            for j in range(k + 1):
                Bbar[k * self.nx : (k + 1) * self.nx, j * self.nu : (j + 1) * self.nu] = (
                    np.linalg.matrix_power(self.A, k - j - 1) @ self.B
                )

        # TODO note H is independent of x0 and Yd, and is thus constant at each
        # step
        H = Rbar + Bbar.T @ Qbar @ Bbar
        g = (Abar @ x0 - xs_des.flatten()).T @ Qbar @ Bbar
        return H, g


class PID(object):
    """PID controller."""

    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        # previous error (for derivative term)
        self.err_prev = 0

        # integral error
        self.err_int = 0

    def control(self, err, dt):
        """Generate control signal u to drive e to zero."""
        derr = (err - self.err_prev) / dt
        self.err_prev = err
        self.err_int += dt * err
        u = self.Kp * err + self.Kd * derr + self.Ki * self.err_int
        return np.atleast_1d(u)


def step(t):
    """Desired trajectory"""
    # step
    if t < 1.0:
        return [0.0, 0]
    return [1.0, 0]


def main():
    nx = 2  # state dimension
    nu = 1  # input dimension

    N = 10  # number of lookahead steps

    dt = 0.1
    tf = 10.0
    num_steps = int(tf / dt)

    pid = PID(Kp=1, Ki=0.1, Kd=1)

    # x+ = Ax + Bu
    # y  = Cx
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0], [dt]])

    sys = LinearSystem(A, B)

    # V = x'Qx + u'Ru
    Q = np.diag([10, 0])
    R = np.eye(nu)
    lb = np.ones(N * nu) * -1.0
    ub = np.ones(N * nu) * 1.0

    t = np.zeros(num_steps)
    xs_des = np.zeros((num_steps, nx))
    xs_pid = np.zeros((num_steps, nx))
    xs_mpc = np.zeros((num_steps, nx))

    U = cp.Variable(nu * N)

    for i in range(num_steps - 1):
        # desired trajectory
        xs_des[i, :] = step(t[i])

        # PID control
        err = xs_des[i, 0] - xs_pid[i, 0]
        u = pid.control(err, dt)
        xs_pid[i + 1, :] = sys.step(xs_pid[i, :], u)

        # MPC (single-shooting)
        xs_des_horizon = np.array([step(t[i] + dt * j) for j in range(N)])
        H, g = sys.lookahead(xs_mpc[i, :], xs_des_horizon, Q, R)

        objective = cp.Minimize(0.5 * cp.quad_form(U, H) + g @ U)
        constraints = [U >= lb, U <= ub]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        u = U.value[:sys.nu]

        xs_mpc[i + 1, :] = sys.step(xs_mpc[i, :], u)
        t[i + 1] = t[i] + dt

    plt.subplot(211)
    plt.plot(t, xs_des[:, 0], label="desired")
    plt.plot(t, xs_pid[:, 0], label="actual")
    plt.title("PID control")
    plt.legend()
    plt.grid()

    plt.subplot(212)
    plt.plot(t, xs_des[:, 0], label="desired")
    plt.plot(t, xs_mpc[:, 0], label="actual")
    plt.title("MPC (single shooting)")
    plt.legend()
    plt.grid()
    plt.xlabel("Time (s)")

    plt.show()


if __name__ == "__main__":
    main()
