#!/usr/bin/env python3

# Simulation of comparison between PID and MPC for linear systems.

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


class LinearSystem:
    """Linear system of the form
    x_{k+1} = A*x_k + B*u_k
    y_k     = C*x_k"""

    def __init__(self, A, B, lb, ub):
        self.A = A
        self.B = B
        self.lb = lb
        self.ub = ub

        self.nx = A.shape[0]  # state dimension
        self.nu = B.shape[1]  # input dimension

    def step(self, x, u):
        u = np.minimum(u, self.ub)
        u = np.maximum(u, self.lb)
        return self.A @ x + self.B @ u


class SingleShootingLinearMPC:
    """Linear MPC using single shooting."""
    def __init__(self, sys, N, Q, R):
        self.sys = sys
        self.N = N

        I = np.eye(N)
        self.Qbar = np.kron(I, Q)
        self.Rbar = np.kron(I, R)
        self.lb = np.kron(np.ones(N), sys.lb)
        self.ub = np.kron(np.ones(N), sys.ub)

        # TODO right now we don't include a separate QN for terminal condition
        self.Abar = np.vstack([np.linalg.matrix_power(sys.A, k + 1) for k in range(N)])
        self.Bbar = np.zeros((sys.nx * N, sys.nu * N))
        for k in range(N):
            sx = np.s_[k * sys.nx : (k + 1) * sys.nx]
            for j in range(k + 1):
                self.Bbar[sx, j * sys.nu : (j + 1) * sys.nu] = (
                    np.linalg.matrix_power(sys.A, k - j - 1) @ sys.B
                )

        self.H = self.Rbar + self.Bbar.T @ self.Qbar @ self.Bbar

    def control(self, x, xs_des):
        g = (self.Abar @ x - xs_des.flatten()).T @ self.Qbar @ self.Bbar

        U = cp.Variable(self.sys.nu * self.N)
        objective = cp.Minimize(0.5 * cp.quad_form(U, self.H) + g @ U)
        constraints = [U >= self.lb, U <= self.ub]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return U.value[: self.sys.nu]


class MultiShootingLinearMPC:
    """Linear MPC using multiple shooting."""
    def __init__(self, sys, N, Q, R):
        self.sys = sys
        self.N = N

        I = np.eye(N)
        self.Qbar = np.kron(I, Q)
        self.Rbar = np.kron(I, R)
        self.Bbar = np.kron(I, sys.B)
        self.Abar = np.kron(I, sys.A)
        self.lb = np.kron(np.ones(N), sys.lb)
        self.ub = np.kron(np.ones(N), sys.ub)

    def control(self, x, xs_des):
        U = cp.Variable(self.sys.nu * self.N)
        X = cp.Variable(self.sys.nx * self.N)

        objective = cp.Minimize(
            0.5 * cp.quad_form(X - xs_des.flatten(), self.Qbar)
            + 0.5 * cp.quad_form(U, self.Rbar)
        )
        X0 = cp.hstack((x, X[: -self.sys.nx]))
        constraints = [X == self.Abar @ X0 + self.Bbar @ U, U >= self.lb, U <= self.ub]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return U.value[: self.sys.nu]


class PID:
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
    num_steps = int(tf / dt) + 1

    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0], [dt]])
    ub = np.ones(nu)
    lb = -ub
    sys = LinearSystem(A, B, lb, ub)

    # controllers
    pid = PID(Kp=1, Ki=0.1, Kd=1)

    Q = np.diag([10, 0])
    R = np.eye(nu)
    mpc_ss = SingleShootingLinearMPC(sys=sys, N=N, Q=Q, R=R)
    mpc_ms = MultiShootingLinearMPC(sys=sys, N=N, Q=Q, R=R)

    ts = dt * np.arange(num_steps)
    xs_des = np.array([step(t) for t in ts])

    xs_pid = np.zeros((num_steps, nx))
    xs_ss = np.zeros((num_steps, nx))
    xs_ms = np.zeros((num_steps, nx))

    for i in range(num_steps - 1):
        # PID control
        err = xs_des[i, 0] - xs_pid[i, 0]
        u = pid.control(err, dt)
        xs_pid[i + 1, :] = sys.step(xs_pid[i, :], u)

        # rollout the desired trajectory for MPC
        xs_des_horizon = np.array([step(ts[i] + dt * j) for j in range(N)])

        # MPC (single-shooting)
        u = mpc_ss.control(xs_ss[i, :], xs_des_horizon)
        xs_ss[i + 1, :] = sys.step(xs_ss[i, :], u)

        # MPC (multi-shooting)
        u = mpc_ms.control(xs_ms[i, :], xs_des_horizon)
        xs_ms[i + 1, :] = sys.step(xs_ms[i, :], u)

    fig = plt.figure()

    plt.subplot(311)
    plt.plot(ts, xs_des[:, 0], label="desired")
    plt.plot(ts, xs_pid[:, 0], label="actual")
    plt.title("PID control")
    plt.legend()
    plt.grid()

    plt.subplot(312)
    plt.plot(ts, xs_des[:, 0], label="desired")
    plt.plot(ts, xs_ss[:, 0], label="actual")
    plt.title("MPC (single shooting)")
    plt.legend()
    plt.grid()
    plt.xlabel("Time (s)")

    plt.subplot(313)
    plt.plot(ts, xs_des[:, 0], label="desired")
    plt.plot(ts, xs_ms[:, 0], label="actual")
    plt.title("MPC (multi shooting)")
    plt.legend()
    plt.grid()
    plt.xlabel("Time (s)")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
