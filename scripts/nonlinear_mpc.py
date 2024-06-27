#!/usr/bin/env python3
"""This needs to be revised."""

import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import cvxpy as cp

import IPython


NUM_ITER = 1  # number of relinearizations


class Pendulum:
    """Nonlinear single pendulum system.

    The state is x = [θ, \dot{θ}], where θ is the pendulum's angle. θ = 0
    corresponds to the pendulum hanging downward, and θ increases in the
    counter-clockwise direction.

    The input u is the torque applied at the base of the pendulum.
    """

    def __init__(self, dt, lb, ub):
        self.dt = dt
        self.g = -1.0  # gravity
        self.l = 1.0  # length
        self.m = 1.0  # mass

        self.nx = 2  # state dimension
        self.nu = 1  # input dimension

        self.lb = lb
        self.ub = ub

    def step(self, x, u):
        """Motion model: x_k+1 = f(x_k, u_k)"""
        u = np.minimum(u, self.ub)
        u = np.maximum(u, self.lb)
        α = self.g * np.sin(x[0]) / self.l + u[0] / (self.m * self.l**2)
        return x + self.dt * np.array([x[1], α])

    def A(self, x):
        """Linearized A matrix."""
        return np.array([[1, self.dt], [self.dt * self.g * np.cos(x[0]) / self.l, 1]])

    def B(self, x):
        """Linearized B matrix."""
        return np.array([[0], [1]])


class NonlinearMPC:
    """Nonlinear model predictive controller."""

    def __init__(self, sys, N, Q, R):
        self.sys = sys
        self.N = N

        assert Q.shape == (sys.nx, sys.nx)
        assert R.shape == (sys.nu, sys.nu)

        self.Qbar = np.kron(np.eye(N + 1), Q)
        self.Rbar = np.kron(np.eye(N), R)
        self.lb = np.kron(np.ones(N), sys.lb)
        self.ub = np.kron(np.ones(N), sys.ub)

    def _rollout(self, x, us):
        """Rollout the state into the future.

        Parameters
        ----------
        x : np.ndarray
            The current state vector.
        us : np.ndarray
            The input trajectory.

        Returns
        -------
        :
            The state trajectory rolled out from ``x`` using the inputs ``us``.
        """

        # rollout given the current policy (inputs)
        xs = np.zeros((self.N + 1, self.sys.nx))
        xs[0, :] = x
        for k in range(self.N):
            xs[k + 1, :] = self.sys.step(xs[k, :], us[k, :])
        return xs

    def _linearize(self, xs):
        Abar = np.zeros((self.sys.nx * self.N, self.sys.nx * (self.N + 1)))
        Abar[:, : -self.sys.nx] = block_diag(*[self.sys.A(x) for x in xs[:-1, :]])
        Abar[:, self.sys.nx :] += -np.eye(self.sys.nx * self.N)

        Bbar = block_diag(*[self.sys.B(x) for x in xs[:-1, :]])

        return Abar, Bbar

    def _solve_qp(self, x, xs_des, us_prev):
        xs_prev = self._rollout(x, us_prev)
        Abar, Bbar = self._linearize(xs_prev)

        U_prev = us_prev.flatten()
        X_prev = xs_prev.flatten()
        Xd = xs_des.flatten()

        δU = cp.Variable(self.N * self.sys.nu)
        δX = cp.Variable((self.N + 1) * self.sys.nx)

        state_cost = 0.5 * cp.quad_form(δX, self.Qbar) + (X_prev - Xd) @ self.Qbar @ δX
        input_cost = 0.5 * cp.quad_form(δU, self.Rbar) + U_prev @ self.Rbar @ δU

        objective = cp.Minimize(state_cost + input_cost)
        constraints = [
            δX[: self.sys.nx] == 0,
            Abar @ (δX + X_prev) + Bbar @ (δU + U_prev) == 0,
            δU >= self.lb - U_prev,
            δU <= self.ub - U_prev,
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return δU.value.reshape((self.N, self.sys.nu))

    def solve(self, x, xs_des, us=None):
        """Solve the MPC problem at current state x given desired state
        trajectory xs_des."""
        # initialize optimal inputs
        if us is None:
            us = np.zeros((self.N, self.sys.nu))

        # iterate the QP
        for i in range(NUM_ITER):
            δus = self._solve_qp(x, xs_des, us)
            us = us + δus

        # return first optimal input
        return us[0, :]


def desired_state(t):
    """Desired trajectory"""
    # step
    if t < 1.0:
        return [0.0, 0]
    return [np.pi, 0]


def main():
    N = 20
    dt = 0.1
    tf = 10.0
    num_steps = int(tf / dt) + 1

    lb = -10.0
    ub = 10.0
    sys = Pendulum(dt, lb, ub)

    Q = np.diag([10, 0])
    R = np.eye(sys.nu) * 0.1
    mpc = NonlinearMPC(sys, N, Q, R)

    ts = np.array([i * dt for i in range(num_steps)])
    xs = np.zeros((num_steps, sys.nx))
    us = np.zeros((num_steps, sys.nu))

    # desired trajectory
    xs_des = np.array([desired_state(t) for t in ts])

    # simulate
    for i in range(num_steps - 1):
        xs_des_horizon = np.array([desired_state(ts[i] + dt * j) for j in range(N + 1)])
        u = mpc.solve(xs[i, :], xs_des_horizon)

        us[i, :] = u
        xs[i + 1, :] = sys.step(xs[i, :], u)

    plt.plot(ts, xs_des[:, 0], label="$\\theta_d$", color="k", linestyle="--")
    plt.plot(ts, xs[:, 0], label="$\\theta$")
    plt.plot(ts, xs[:, 1], label="$\dot{\\theta}$")
    plt.plot(ts, us, label="$u$")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
