from casadi import *
import numpy as np
import scipy.linalg as lin
from scipy import signal
from Models import CarModel


class LQR:
    def __init__(self):
        self.model = CarModel()
        self.Q = np.diag([1, 1, 1])
        self.R = 1 * np.eye(self.model.nu)

        self.P = 0
        self.B = 0
        self.K = 0

        # Discrete system matrices
        self.A_d = 0.0
        self.B_d = 0.0

    def lqr(self):

        x = SX.sym('x', self.model.nx, 1)
        u = SX.sym('u', self.model.nu, 1)
        xdot = self.model.system(x, u)
        nx = self.model.nx
        nu = self.model.nu

        A_fun = Function('Linearized_A', [x, u], [jacobian(xdot, x)])
        B_fun = Function('Linearized_B', [x, u], [jacobian(xdot, u)])

        xs = np.ones([nx, 1])
        us = np.zeros([nu, 1])

        A = A_fun(xs, us)
        B = B_fun(xs, us)
        C = np.eye(nx)
        D = np.zeros([nx, nu])

        sys = signal.StateSpace(A, B, C, D)
        sys_d = sys.to_discrete(dt=self.model.h)
        self.A_d = sys_d.A
        self.B_d = sys_d.B

        self.P = lin.solve_discrete_are(self.A_d, self.B_d, self.Q, self.R)

        return self.P, self.B_d

    def step(self, x):
        self. P, self.B = self.lqr()
        self.K = 1/self.R * self.B.T @ self.P
        u = - np.dot(self.K, x)

        if u < -23:
            u = -23
        elif u > 23:
            u = 23

        return u

