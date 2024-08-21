#!/usr/bin/env python3
import sys
import numpy as np
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
from Helper_scripts.Models import *
from casadi import *


class OCP:
    def __init__(self, model_type, horizon, shooting_nodes, track_length, track_width, x0, constrained=True,
                 weights=None, ocp_type="MPCC"):
        # OCP parameters
        self.Tf = horizon
        self.N = shooting_nodes
        self.t_samp = (self.Tf/self.N)
        self.constrained = constrained
        self.ocp_type = ocp_type

        self.model_type = model_type

        if weights is not None:
            self.qc = weights[0]
            self.ql = weights[1]
            self.qtheta = weights[2]

        # Track parameters
        self.track_len = track_length
        self.track_width = track_width

        # System parameters
        self.nx = None
        self.nu = None
        self.x0 = x0

        self.system = None
        self.init_system = None

    def build(self, n_params, initial=False):

        ocp = AcadosOcp()

        if self.model_type.casefold() == 'dynamic'.casefold():
            self.system = CarModel(model_typ=self.model_type)
            model = self.system.Dynamic_model(track_length=self.track_len)

        elif self.model_type.casefold() == 'bicycle'.casefold():
            if initial:
                self.init_system = CarModel(model_typ=self.model_type, initial=initial)
                model = self.init_system.Bicycle_Model(track_length=self.track_len)
            else:
                self.system = CarModel(model_typ=self.model_type, initial=initial)
                model = self.system.Bicycle_Model(track_length=self.track_len)
        else:
            self.system = CarModel(model_typ=self.model_type)
            model = self.system.kinematic_bicycle_model(track_length=self.track_len)

        # Create ACADOS Model
        model_ac = AcadosModel()
        model_ac.name = model.name
        model_ac.f_impl_expr = model.f_impl_expr
        model_ac.f_expl_expr = model.f_expl_expr
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        ocp.model = model_ac  # Assign OCP model as ACADOS model

        """ SETTING PARAMETERS """
        # Initialize parameter values
        model_ac.p = SX.sym('p', n_params)
        ocp.parameter_values = np.zeros(n_params)

        """ COST FUNCTION """
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        if self.model_type.casefold() == 'dynamic'.casefold():
            pose = vertcat(model_ac.x[0], model_ac.x[1], model_ac.x[2])
        elif self.model_type.casefold() == 'bicycle'.casefold():
            pose = vertcat(model_ac.x[0], model_ac.x[1], model_ac.x[4])
        else:
            pose = model_ac.x[:3]

        X_k = pose[0]
        Y_k = pose[1]
        psi_k = pose[2]
        if initial:
            theta_k = model_ac.x[self.init_system.nx-1]
        else:
            theta_k = model_ac.x[self.system.nx - 1]

        v_theta_k = model_ac.u[2]

        ref_pose = model_ac.p[:3]
        X_ref = ref_pose[0]
        Y_ref = ref_pose[1]
        psi_ref = ref_pose[2]

        e_c = sin(psi_ref)*(X_k-X_ref) - cos(psi_ref)*(Y_k-Y_ref)
        e_l = -cos(psi_ref)*(X_k-X_ref) - sin(psi_ref)*(Y_k-Y_ref)

        model.cost_expr = self.qc*(e_c**2) + self.ql*(e_l**2) - self.qtheta*v_theta_k*self.t_samp
        model.cost_expr_e = -self.qtheta*theta_k
        # model.cost_Fcn = Function("Cost_Function", [pose, ref_pose], [model.cost_expr])

        model.ec = Function('Contouring_Error', [pose, ref_pose], [e_c])
        model.el = Function('Lateral_Error', [pose, ref_pose], [e_l])

        ocp.model.cost_expr_ext_cost = model.cost_expr
        ocp.model.cost_expr_ext_cost_e = model.cost_expr_e

        """ STATE CONSTRAINTS """
        if self.constrained:
            constraint = types.SimpleNamespace()

            # First shooting node
            ocp.constraints.x0 = self.x0

            # Track bounds
            # Polyhedral constraints for track bounds
            dmax = self.track_width/2

            if n_params > 3:
                x_l = model_ac.p[3]
                y_l = model_ac.p[4]
                x_r = model_ac.p[5]
                y_r = model_ac.p[6]
                m = (x_r - x_l)/(y_l - y_r)

                f1 = (y_l - m * x_l)
                f2 = (y_r - m * x_r)

                tr_min = if_else(f1 <= f2, f1, f2)
                tr_max = if_else(f1 >= f2, f1, f2)
                constraint.expr = Y_k - m*X_k

                ocp.model.con_h_expr_0 = constraint.expr
                ocp.constraints.lh_0 = np.array([tr_min])
                ocp.constraints.uh_0 = np.array([tr_max])

                ocp.model.con_h_expr = constraint.expr
                ocp.constraints.lh = np.array([tr_min])
                ocp.constraints.uh = np.array([tr_max])

                ocp.model.con_h_expr_e = constraint.expr
                ocp.constraints.lh_e = np.array([tr_min])
                ocp.constraints.uh_e = np.array([tr_max])

            # Contouring Error as track bound constraint
            else:
                constraint.expr = sin(psi_ref)*(X_k-X_ref) - cos(psi_ref)*(Y_k-Y_ref)  # e_c

                ocp.model.con_h_expr_0 = constraint.expr
                ocp.constraints.lh_0 = np.array([-dmax])  # -dmax
                ocp.constraints.uh_0 = np.array([dmax])

                ocp.model.con_h_expr = constraint.expr
                ocp.constraints.lh = np.array([-dmax])
                ocp.constraints.uh = np.array([dmax])

                ocp.model.con_h_expr_e = constraint.expr
                ocp.constraints.lh_e = np.array([-dmax])
                ocp.constraints.uh_e = np.array([dmax])

        else:
            ocp.constraints.x0 = self.x0

        # Other state constraints
        ocp.constraints.lbx = model.xmin
        ocp.constraints.ubx = model.xmax
        ocp.constraints.idxbx = model.x_idx

        ocp.constraints.lbx_e = model.xmin
        ocp.constraints.ubx_e = model.xmax
        ocp.constraints.idxbx_e = model.x_idx

        """ INPUT CONSTRAINTS """
        # Inputs: [vx, delta, v_theta]
        ocp.constraints.lbu = model.umin
        ocp.constraints.ubu = model.umax
        ocp.constraints.idxbu = np.array([0, 1, 2])

        """ SOLVER OPTIONS """
        ocp.solver_options.tf = self.Tf
        ocp.dims.N = self.N

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.integrator_type = "IRK"
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_max_iter = 300
        ocp.solver_options.tol = 1e-1
        ocp.solver_options.regularize_method = 'MIRROR'

        ocp.code_export_directory = 'ACADOS_OCP'
        solver = AcadosOcpSolver(ocp, json_file=f"{self.ocp_type}_{self.model_type}_OCP.json")

        return solver


class MPC:
    def __init__(self, model_type, horizon, shooting_nodes, track_length, track_width, x0, weights, n_params,
                 constrained=True):

        self.ocp = OCP(horizon=horizon, shooting_nodes=shooting_nodes, track_length=track_length,
                       track_width=track_width, x0=x0, weights=weights, model_type=model_type, constrained=constrained)

        if model_type.casefold() == 'Kinematic Bicycle'.casefold() or model_type.casefold() == 'Dynamic'.casefold():
            self.ocp_solver = self.ocp.build(n_params=n_params)

        elif model_type.casefold() == 'Bicycle'.casefold():
            self.ocp_solver_init = self.ocp.build(initial=True, n_params=n_params)
            self.ocp_solver = self.ocp.build(initial=False, n_params=n_params)

        else:
            print('Please enter valid model type')
            sys.exit()

        self.nx = self.ocp.system.nx
        self.nu = self.ocp.system.nu
        self.N = shooting_nodes
        self.model_type = model_type

        self.opt_xi = None
        self.opt_ui = None

        self.opt_x = np.array([])
        self.opt_u = np.array([])

        self.init_x = None
        self.init_u = None

    def solve_ocp(self, solver, x, x_init, u_init, param, iteration, ref_idx=None, n_params=3):

        for j in range(self.N):
            if j == 0:
                # Set current state as initial state at first shooting node
                self.set_init_state(solver=solver, state=x)
            if ref_idx is None:
                solver.set(j, 'p', param[j, :n_params])
            else:
                solver.set_params_sparse(j, ref_idx, param[j, :n_params])

        if ref_idx is None:
            solver.set(self.N, 'p', param[self.N, :n_params])
        else:
            solver.set_params_sparse(self.N, ref_idx, param[self.N, :n_params])

        self.set_init_guess(solver=solver, init_x=x_init, init_u=u_init)

        # SOLVE OCP
        status = solver.solve()  # Solve the OCP

        if status != 0:
            print("acados returned status {} in closed loop iteration {}.".format(status, iteration))
            solver.print_statistics()
        else:
            pass
            print("Optimal solution found")

        self.opt_xi, self.opt_ui = self.get_solution(solver, iteration)

        return self.opt_xi, self.opt_ui, status

    def get_solution(self, solver, iteration):

        # Reinitialize to store solution of current iteration
        self.opt_xi = []
        self.opt_ui = []

        for k in range(self.N + 1):
            optx_k = solver.get(k, "x")
            self.opt_xi = np.concatenate((self.opt_xi, optx_k))

        for k in range(self.N):
            optu_k = solver.get(k, "u")
            self.opt_ui = np.concatenate((self.opt_ui, optu_k))

        self.opt_xi = self.opt_xi.reshape(self.N + 1, self.nx)  # Optimal state trajectory at iteration i
        self.opt_ui = self.opt_ui.reshape(self.N, self.nu)  # Optimal input trajectory at iteration i

        if iteration == 0:
            self.opt_x = np.append(self.opt_x, self.opt_xi).reshape(self.N+1, self.nx)
            self.opt_u = np.append(self.opt_u, self.opt_ui).reshape(self.N, self.nu)
        else:
            self.opt_x = np.vstack((self.opt_x, self.opt_xi))
            self.opt_u = np.vstack((self.opt_u, self.opt_ui))

        return self.opt_xi, self.opt_ui

    def generate_init_guess(self, x0, initial=True):

        if self.model_type.casefold() == 'dynamic'.casefold():
            u1 = np.ones(self.N)  # Velocity
            u2 = np.zeros(self.N)  # Steering angle
            u3 = np.ones(self.N)  # Virtual speed/ Projected speed

            self.init_u = np.column_stack((u1, u2, u3))
            self.init_x = np.array(x0)
            x = x0
            for u in self.init_u:
                x = self.ocp.system.simulate(x,u)
                self.init_x = np.vstack((self.init_x, x.T))

        elif self.model_type.casefold() == 'bicycle'.casefold():
            u1 = np.zeros(self.N)  # Steering angle rate
            u2 = np.ones(self.N)  # Longitudinal acceleration
            u3 = np.ones(self.N)  # Virtual speed/ Projected speed

            self.init_u = np.column_stack((u1, u2, u3))
            self.init_x = np.array([x0])
            x = x0
            if initial:
                for u in self.init_u:
                    x = self.ocp.init_system.simulate(x, u)
                    self.init_x = np.vstack((self.init_x, x.T))
            else:
                for u in self.init_u:
                    x = self.ocp.system.simulate(x, u)
                    self.init_x = np.vstack((self.init_x, x.T))
        else:
            u1 = np.zeros(self.N)  # Steering angle
            u2 = np.ones(self.N)  # Velocity
            u3 = np.ones(self.N)  # Virtual speed/ Projected speed

            self.init_u = np.column_stack((u1, u2, u3))

            self.init_x = np.array([x0])
            x = x0
            for u in self.init_u:
                x = self.ocp.system.simulate(x, u)
                self.init_x = np.vstack((self.init_x, x.T))

        return self.init_x, self.init_u

    def warm_start(self, t_samp):

        if (self.model_type.casefold() == 'Bicycle'.casefold() or
                self.model_type.casefold() == 'dynamic'.casefold()):

            xf = self.opt_xi[-1, :]
            uf = self.opt_ui[-1, :]

            xN_1 = self.ocp.system.simulate(xf, uf).reshape(1, xf.shape[0])

            self.init_x = np.vstack((self.opt_xi[1:, :], xN_1))
            self.init_u = np.vstack((self.opt_ui[1:, :], uf))

        else:

            xf = self.opt_xi[-1, :]
            uf = self.opt_ui[-1, :]

            xN_1 = self.ocp.system.simulate(xf, uf)

            self.init_x = np.vstack((self.opt_xi[1:, :], xN_1.T))
            self.init_u = np.vstack((self.opt_ui[1:, :], uf))

            if not self.ocp.system.aug:
                xf = self.opt_xi[-1, :3]
                uf = self.opt_ui[-1, :2]

                xN_1 = self.ocp.system.simulate(xf, uf)

                self.init_x = np.vstack((self.opt_xi[1:, :3], xN_1))
                self.init_u = np.vstack((self.opt_ui[1:, :2], uf))

        return self.init_x, self.init_u

    def set_init_guess(self, solver, init_x, init_u):

        for i in range(self.N + 1):
            solver.set(i, 'x', init_x[i, :])

        for i in range(self.N):
            solver.set(i, 'u', init_u[i, :])

    @staticmethod
    def set_init_state(solver, state):

        solver.set(0, 'lbx', state)
        solver.set(0, 'ubx', state)


def ec(pose, ref_pose):
    X_k = pose[0]
    Y_k = pose[1]

    X_ref = ref_pose[0]
    Y_ref = ref_pose[1]
    psi_ref = ref_pose[2]
    return np.sin(psi_ref) * (X_k - X_ref) - np.cos(psi_ref) * (Y_k - Y_ref)


def el(pose, ref_pose):
    X_k = pose[0]
    Y_k = pose[1]

    X_ref = ref_pose[0]
    Y_ref = ref_pose[1]
    psi_ref = ref_pose[2]
    return -np.cos(psi_ref) * (X_k - X_ref) - np.sin(psi_ref) * (Y_k - Y_ref)


def ContouringCost(x, xref, q_c, q_l, q_theta):

    # Contouring and Lag error terms
    e_c = sin(xref[2])*(x[0]-xref[0]) - cos(xref[2])*(x[1]-xref[1])
    e_l = -cos(xref[2])*(x[0]-xref[0]) - sin(xref[2])*(x[1]-xref[1])
    e = vertcat(e_c, e_l)

    # Weight matrix for each error term
    Q = np.diag([q_c, q_l])

    # Cost function
    cost = e.T @ Q @ e - q_theta * x[3]
    return cost


# def TrajectoryTrackingOCP(self):
    #
    #     ocp = AcadosOcp()
    #     model = kinematic_bicycle_model(augmented=False)
    #     constraint = types.SimpleNamespace()
    #
    #     # Model
    #     model_ac = AcadosModel()
    #     model_ac.name = model.name
    #     model_ac.f_impl_expr = model.f_impl_expr
    #     model_ac.f_expl_expr = model.f_expl_expr
    #     model_ac.x = model.x
    #     model_ac.xdot = model.xdot
    #     model_ac.u = model.u
    #
    #     # set dimensions
    #     nx = model.x.size()[0]
    #     nu = model.u.size()[0]
    #     ny = nx + nu
    #     ny_e = nx
    #
    #     """ OCP SETTINGS """
    #     ocp.dims.N = self.N
    #     Q = np.diag([0.1, 0.1, 1])
    #     R = np.diag([1, 1])
    #     Qt = Q
    #
    #     ocp.cost.cost_type = 'LINEAR_LS'
    #     ocp.cost.cost_type_e = 'LINEAR_LS'
    #
    #     # Stage Cost
    #     ocp.cost.W = lin.block_diag(Q, R)
    #     ocp.cost.Vx = np.zeros((ny, nx))
    #     ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    #     ocp.cost.Vu = np.zeros((ny, nu))
    #     ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
    #
    #     # Terminal Cost
    #     ocp.cost.W_e = Qt
    #     ocp.cost.Vx_e = np.eye(nx)
    #
    #     # set intial references
    #     # st_ref = SX.sym('x_ref', 3)
    #     #
    #     # # ocp.cost.yref = vertcat(st_ref, np.zeros(nu))
    #     # # ocp.cost.yref_e = st_ref
    #     ocp.cost.yref = np.zeros((ny,))
    #     ocp.cost.yref_e = np.zeros((ny_e,))
    #
    #     # State constraints
    #
    #     #   Constraint at initial shooting node/Initial state
    #     ocp.constraints.x0 = np.array([0.0, 0.0, 0.0])
    #
    #     #   Constraint at other shooting nodes
    #     ocp.constraints.lbx = np.array([-500, -500])
    #     ocp.constraints.ubx = np.array([500, 500])
    #     ocp.constraints.idxbx = np.array([0, 1])
    #
    #     # st = model_ac.x
    #     # y_ref = ocp.cost.yref
    #     # constraint.expr = (st[0] - y_ref[0])*sin(y_ref[2]) - (st[1] - y_ref[1])*cos(y_ref[2])
    #     # model_ac.con_h_expr = constraint.
    #     # ocp.constraints.lh = np.array([-1.1])
    #     # ocp.constraints.uh = np.array([1.1])
    #
    #     #   Constraint at final shooting node
    #     ocp.constraints.lbx_e = np.array([-500, -500])
    #     ocp.constraints.ubx_e = np.array([500, 500])
    #     ocp.constraints.idxbx_e = np.array([0, 1])
    #
    #     # Input constraints
    #     ocp.constraints.lbu = np.array([-np.pi, 0.0])
    #     ocp.constraints.ubu = np.array([np.pi, 7.50])
    #     ocp.constraints.idxbu = np.array([0, 1])
    #
    #     # OCP Model
    #     ocp.model = model_ac
    #
    #     # Solver options
    #     ocp.solver_options.tf = self.Tf
    #     ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    #     ocp.solver_options.nlp_solver_type = "SQP"
    #     ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    #     ocp.solver_options.integrator_type = "IRK"
    #     ocp.solver_options.print_level = 0
    #     ocp.solver_options.nlp_solver_max_iter = 300
    #     # ocp.solver_options.qp_solver_iter_max = 200
    #     ocp.solver_options.tol = 1e-3
    #     # ocp.solver_options.qp_solver_cond_N = N
    #
    #     # create solver
    #     acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
    #
    #     return model, constraint, acados_solver
    # def ContouringControlOCP(self, start_pose, track_length, constr_type='Contouring error', constrained=True, dmax=1.0):
    #
    #     ocp = AcadosOcp()
    #     model = kinematic_bicycle_model(start_pose=start_pose)
    #     constraint = types.SimpleNamespace()
    #     constraint.constraint_type = constr_type
    #
    #     # Create ACADOS Model
    #     model_ac = AcadosModel()
    #     model_ac.name = model.name
    #     model_ac.f_impl_expr = model.f_impl_expr
    #     model_ac.f_expl_expr = model.f_expl_expr
    #     model_ac.x = model.x
    #     model_ac.xdot = model.xdot
    #     model_ac.u = model.u
    #
    #     """ SETTING PARAMETERS """
    #     n_param = 8
    #     model_ac.p = SX.sym('p', n_param)
    #
    #     # Initialize parameter values
    #     ocp.parameter_values = np.zeros(n_param)
    #
    #     ocp.model = model_ac  # Assign OCP model as ACADOS model
    #
    #     nx = model.x.size()[0]
    #     nu = model.u.size()[0]
    #
    #     """ COST FUNCTION """
    #     ocp.cost.cost_type = 'EXTERNAL'
    #     ocp.cost.cost_type_e = 'EXTERNAL'
    #
    #     states = model_ac.x
    #     states_ref = model_ac.p[:4]
    #     inputs = model_ac.u
    #
    #     qc = 1.0
    #     ql = 1000.0
    #     qtheta = 1.0
    #
    #     # e_c = sin(states[2]) * (states[0] - states_ref[0]) - cos(states_ref[2]) * (states[1] - states_ref[1])
    #     # e_l = -cos(states[2]) * (states[0] - states_ref[0]) - sin(states_ref[2]) * (states[1] - states_ref[1])
    #
    #     # ocp.model.cost_expr_ext_cost_0 = -qtheta * inputs[2] * t_samp
    #
    #     # ocp.model.cost_expr_ext_cost_0 = qc * (e_c**2) + ql * (e_l**2) - qtheta * inputs[2]*t_samp
    #     # ocp.model.cost_expr_ext_cost = qc * (e_c**2) + ql * (e_l**2) - qtheta * inputs[2]*t_samp
    #     # ocp.model.cost_expr_ext_cost_e = 0.0
    #
    #     ocp.model.cost_expr_ext_cost = self.ContouringCost(states, states_ref, q_c=qc, q_l=ql, q_theta=qtheta)
    #     ocp.model.cost_expr_ext_cost_e = self.ContouringCost(states, states_ref, q_c=qc, q_l=ql, q_theta=qtheta)
    #     # ocp.model.cost_expr_ext_cost_e = 0.0  # -qtheta*states
    #
    #
    #
    #         # elif constraint.constraint_type.casefold() == 'Polyhedral modified'.casefold():
    #         #
    #         #     # Track bounds as polyhedral constraints
    #         #     x_l = model_ac.p[4]
    #         #     y_l = model_ac.p[5]
    #         #     x_r = model_ac.p[6]
    #         #     y_r = model_ac.p[7]
    #         #
    #         #     m = tan(states_ref[2])
    #         #     f1 = m * states[0] + (y_l - m * x_l)
    #         #     f2 = m * states[0] + (y_r - m * x_r)
    #         #
    #         #     tr_min = if_else(f1 <= f2, f1, f2)
    #         #     tr_max = if_else(f1 >= f2, f1, f2)
    #         #     constraint.expr = states[1]
    #         #
    #         #     ocp.model.con_h_expr_0 = constraint.expr
    #         #     ocp.constraints.lh_0 = np.array([tr_min])
    #         #     ocp.constraints.uh_0 = np.array([tr_max])
    #         #
    #         #     ocp.model.con_h_expr = constraint.expr
    #         #     ocp.constraints.lh = np.array([tr_min])
    #         #     ocp.constraints.uh = np.array([tr_max])
    #         #
    #         #     ocp.model.con_h_expr_e = constraint.expr
    #         #     ocp.constraints.lh_e = np.array([tr_min])
    #         #     ocp.constraints.uh_e = np.array([tr_max])
    #         #
    #         #     constraint.tr_min = Function('track_left', [states, states_ref, x_l, y_l, x_r, y_r], [tr_min])
    #         #     constraint.tr_max = Function('track_right', [states, states_ref, x_l, y_l, x_r, y_r], [tr_max])
    #
    #         # casADi function to define constraints
    #         constraint.function = Function('Constraint_Fcn', [states, states_ref], [constraint.expr])
    #
    #         # Other state constraints
    #         ocp.constraints.lbx = np.array([0])
    #         ocp.constraints.ubx = np.array([track_length])
    #         ocp.constraints.idxbx = np.array([3])
    #
    #         ocp.constraints.lbx_e = np.array([0])
    #         ocp.constraints.ubx_e = np.array([track_length])
    #         ocp.constraints.idxbx_e = np.array([3])
    #
    #     """ INPUT CONSTRAINTS """
    #     ocp.constraints.lbu = np.array([-0.4189/2, 0, 0.0])
    #     ocp.constraints.ubu = np.array([0.4189/2, 4.0, 3.0])
    #     ocp.constraints.idxbu = np.array([0, 1, 2])
    #
    #     """ SOLVER OPTIONS """
    #     ocp.solver_options.tf = self.Tf
    #     ocp.dims.N = self.N
    #
    #     ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    #     ocp.solver_options.nlp_solver_type = "SQP"
    #     ocp.solver_options.hessian_approx = "EXACT"
    #     ocp.solver_options.integrator_type = "IRK"
    #     ocp.solver_options.print_level = 0
    #     ocp.solver_options.nlp_solver_max_iter = 300
    #     ocp.solver_options.tol = 1e-2
    #     ocp.solver_options.regularize_method = 'MIRROR'
    #
    #     ocp.code_export_directory = 'MPCC_OCP'
    #     solver = AcadosOcpSolver(ocp, json_file="MPCC_OCP.json")
    #
    #     return model, constraint, solver
# class OCP:
#     """
#     Class to interact with and define an Optimal Control Problem.
#     """
#
#     def __init__(self, x0, xr, ur, i, horizon=1, f=50):
#
#         self.ocp = AcadosOcp()  # OCP Object to formulate OCP
#         self.ocp.model = bicycle_model()  # Car model
#         self.ocp.code_export_directory = "Code_Trajectory_Tracking_OCP"
#
#         self.nx = self.ocp.model.x.size()[0]
#         self.nu = self.ocp. model.u.size()[0]
#         self.ny = self.nx + self.nu
#
#         self.Q = np.diag([1, 1, 1])  # np.diag([1e1, 1e-4, 1e4, 1e-2])
#         self.R = np.diag([1, 1])  # 1e1 * np.eye(self.nu)
#
#         self.Tf = horizon  # Prediction horizon in seconds
#         self.N = f  # Steps per prediction horizon (discretization frequency)
#         self.counter = 0
#
#         self.xr = xr
#         self.ur = ur
#
#         self.x0 = x0
#         self.current_iteration = i
#
#     def build(self):
#         """
#         This function builds an OCP of the form:
#
#             min(x,u) Σ((x-xr)T Q (x-xr) + (u-ur)T R (u-ur)) + (xN-xrN)T Qt (xN-xrN)
#                 s.t
#                     x_dot = model
#                     xlb <= x <= xub
#                     ulb <= u <= uub
#
#                 x : []
#
#         :return: ocp_solver : The solver object to be used to solve the OCP
#         """
#
#         """ _____________________________________________OCP MODELING_______________________________________________ """
#
#         """ Cost Function """
#         self.ocp.cost.cost_type = 'LINEAR_LS'
#         self.ocp.cost.cost_type_e = 'LINEAR_LS'
#
#         self.ocp.cost.W = lin.block_diag(self.Q, self.R)
#         self.ocp.cost.Vx = np.zeros((self.ny, self.nx))
#         self.ocp.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)
#         self.ocp.cost.Vu = np.zeros((self.ny, self.nu))
#         self.ocp.cost.Vx[-self.nu:, -self.nu:] = np.eye(self.nu)
#
#         for j in range(self.N):
#             xref = self.xr[self.current_iteration+j:]
#             uref = self.ur[:]
#             self.ocp.cost.yref = np.append(xref, uref)
#
#         self.ocp.cost.W_e = self.Q
#         self.ocp.cost.Vx_e = np.eye(self.nx)
#         self.ocp.cost.yref_e = self.xr[-1]
#
#         """ State Constraints """
#
#         # self.ocp.constraints.lbx =
#         # self.ocp.constraints.ubx =
#         # self.ocp.constraints.idxbx =   # Constraint applied to 3rd state (cart position)
#
#         """Input Constraints"""
#         # self.ocp.constraints.lbu = np.array([]
#         # self.ocp.constraints.ubu = np.array([])
#         # self.ocp.constraints.idxbu = np.array([0, 1])
#         # self.ocp.constraints.x0 = self.x0
#         #
#         # """Solver Options"""
#         # self.ocp.dims.N = self.N
#         # self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
#         # self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
#         # self.ocp.solver_options.integrator_type = 'IRK'  # System model integrator
#         # self.ocp.solver_options.print_level = 0
#         # self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
#         # self.ocp.solver_options.qp_solver_cond_N = self.N  # Stages per Prediction horizon
#         # self.ocp.solver_options.tol = 1e-2  # Tolerance
#         # self.ocp.solver_options.nlp_solver_max_iter = 100
#         # self.ocp.solver_options.tf = self.Tf  # Prediction horizon
#         # self.ocp.solver_options.levenberg_marquardt = 10.0
#         #
#         # """ Generate .json file to build solver """
#         # AcadosOcpSolver.generate(self.ocp, json_file='acados_MPC.json')
#         # AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)