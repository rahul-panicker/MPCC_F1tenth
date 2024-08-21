from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
from Helper_scripts.Models import *
from casadi import *
import scipy.linalg as lin


class OCP:
    def __init__(self, horizon, shooting_nodes, track_length, track_width, x0, constrained=True,
                 model_type='kinematic bicycle'):

        # OCP parameters
        self.Tf = horizon
        self.N = shooting_nodes
        self.t_samp = (self.Tf/self.N)
        self.constrained = constrained
        self.ocp_type = None
        self.model_type = model_type
        self.cost_Fcn = None

        # Weights
        self.qc = None
        self.ql = None
        self.qs = None
        self.R = None

        # Track parameters
        self.track_len = track_length
        self.track_width = track_width

        # System parameters
        self.x0 = x0
        self.system = CarModel(model_typ=self.model_type)

    def build_MPCC_OCP(self, n_params, weights=None, R=None):

        ocp = AcadosOcp()

        if self.model_type.casefold() == 'kinematic bicycle'.casefold():
            model = self.system.MPCC_kinematic_model()
        else:
            model = self.system.MPCC_dynamic_model()

        # Create ACADOS Model
        model_ac = AcadosModel()
        model_ac.name = model.name
        model_ac.f_impl_expr = model.f_impl_expr
        model_ac.f_expl_expr = model.f_expl_expr
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        ocp.model = model_ac

        """ SETTING PARAMETERS """
        # Initialize parameter values
        model_ac.p = SX.sym('p', n_params)
        ocp.parameter_values = np.zeros(n_params)

        """ COST FUNCTION """
        if weights is None:
            [self.qc, self.ql, self.qs] = [1e-4, 1, 0.5]  # Default weights
            self.R = np.diag([1e-2, 1e-3, 1e-3])
        else:
            [self.qc, self.ql, self.qs] = weights
            self.R = R

        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        if self.model_type.casefold() == 'kinematic bicycle'.casefold():
            X_k = model_ac.x[0]
            Y_k = model_ac.x[1]
            s_k = model_ac.x[3]
            v_s_k = model_ac.x[6]

            del_u = model_ac.u
        else:
            X_k = model_ac.x[0]
            Y_k = model_ac.x[1]
            s_k = model_ac.x[6]
            v_s_k = model_ac.x[9]
            del_u = model_ac.u

        X_ref = model_ac.p[0]
        Y_ref = model_ac.p[1]
        psi_ref = model_ac.p[2]

        e_c = sin(psi_ref) * (X_k - X_ref) - cos(psi_ref) * (Y_k - Y_ref)
        e_l = -cos(psi_ref) * (X_k - X_ref) - sin(psi_ref) * (Y_k - Y_ref)

        model.cost_expr = self.qc*(e_c**2) + self.ql*(e_l**2) + del_u.T @ self.R @ del_u - self.qs*v_s_k*self.t_samp
        model.cost_expr_e = -self.qs*s_k

        ocp.model.cost_expr_ext_cost = model.cost_expr
        ocp.model.cost_expr_ext_cost_e = model.cost_expr_e

        self.cost_Fcn = Function("Cost_Function", [model_ac.x, model_ac.u, model_ac.p], [model.cost_expr])

        """ STATE CONSTRAINTS """
        # Constraint at first shooting node
        ocp.constraints.x0 = self.x0

        if self.constrained:
            constraint = types.SimpleNamespace()

            dmax = self.track_width/2
            constraint.expr = e_c
            ocp.model.con_h_expr_0 = constraint.expr
            ocp.constraints.lh_0 = np.array([-dmax])
            ocp.constraints.uh_0 = np.array([dmax])

            ocp.model.con_h_expr = constraint.expr
            ocp.constraints.lh = np.array([-dmax])
            ocp.constraints.uh = np.array([dmax])

            ocp.model.con_h_expr_e = constraint.expr
            ocp.constraints.lh_e = np.array([-dmax])
            ocp.constraints.uh_e = np.array([dmax])

        # Other state constraints
        ocp.constraints.lbx = model.xmin
        ocp.constraints.ubx = model.xmax
        ocp.constraints.idxbx = model.x_idx

        ocp.constraints.lbx_e = model.xmin
        ocp.constraints.ubx_e = model.xmax
        ocp.constraints.idxbx_e = model.x_idx

        """ INPUT CONSTRAINTS """
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
        ocp.solver_options.tol = 1e-1  # 1e-3 - Lower solver tolerance for trajectory tracking
        ocp.solver_options.regularize_method = 'MIRROR'

        self.ocp_type = 'MPCC'
        ocp.code_export_directory = f'{self.ocp_type}_Code_Directory'
        solver = AcadosOcpSolver(ocp, json_file=f"{self.ocp_type}_{self.model_type}_OCP.json")

        return solver

    def build_tracking_OCP(self, Q, R):
        ocp = AcadosOcp()

        if self.model_type.casefold() == 'kinematic bicycle'.casefold():
            model = self.system.TrackingMPC_kinematic_model()
        else:
            model = None

        # Create ACADOS Model
        model_ac = AcadosModel()
        model_ac.name = model.name
        model_ac.f_impl_expr = model.f_impl_expr
        model_ac.f_expl_expr = model.f_expl_expr
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        ocp.model = model_ac

        """ COST FUNCTION """
        Q = Q
        R = R
        Qt = Q

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        # Stage Cost
        nx = self.system.nx
        nu = self.system.nu
        ny = nx + nu
        ny_e = nx

        ocp.cost.W = lin.block_diag(Q, R)
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.yref = np.zeros((ny,))

        # Terminal Cost
        ocp.cost.W_e = Qt
        ocp.cost.Vx_e = np.eye(nx)
        ocp.cost.yref_e = np.zeros((ny_e,))

        """CONSTRAINTS"""
        ocp.constraints.x0 = self.x0
        if self.constrained:
            pass
            # Constraints come here

        """SOLVER OPTIONS"""
        ocp.solver_options.tf = self.Tf
        ocp.dims.N = self.N

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "IRK"
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_max_iter = 300
        # ocp.solver_options.qp_solver_iter_max = 200
        ocp.solver_options.tol = 1e-3
        ocp.solver_options.qp_solver_cond_N = self.N

        self.ocp_type = 'Tracking_MPC'
        ocp.code_export_directory = f'{self.ocp_type}_Code_Directory'
        solver = AcadosOcpSolver(ocp, json_file=f"{self.ocp_type}_{self.model_type}_OCP.json")
        return solver


class MPCC:
    def __init__(self, model_type, horizon, shooting_nodes, track, x0, weights, n_params,
                 constrained=True, R=None):

        self.ocp = OCP(horizon=horizon, shooting_nodes=shooting_nodes, track_length=track.length,
                       track_width=track.width, x0=x0, model_type=model_type, constrained=constrained)

        if model_type.casefold() == 'Kinematic Bicycle'.casefold():
            self.ocp_solver = self.ocp.build_MPCC_OCP(n_params=n_params, weights=weights, R=R)
        else:
            self.ocp_solver = self.ocp.build_MPCC_OCP(n_params=n_params, weights=weights, R=R)

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

        self.t_samp = self.ocp.t_samp

    def set_init_guess(self, solver, init_x, init_u):

        for i in range(self.N + 1):
            solver.set(i, 'x', init_x[i, :])

        for i in range(self.N):
            solver.set(i, 'u', init_u[i, :])

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

    @staticmethod
    def set_init_state(solver, state):

        solver.set(0, 'lbx', state)
        solver.set(0, 'ubx', state)

    def solve_ocp(self, solver, x, x_init, u_init, param, iteration, n_params=3):

        # Set parameters for OCP
        for j in range(self.N+1):
            solver.set(j, 'p', param[j, :n_params])
            if j == 0:
                # Set current state as initial state at first shooting node
                self.set_init_state(solver=solver, state=x)

        # Set initial guess
        self.set_init_guess(solver=solver, init_x=x_init, init_u=u_init)

        # Solve OCP
        status = solver.solve()  # Solve the OCP

        if status != 0:
            print("acados returned status {} in closed loop iteration {}.".format(status, iteration))
            solver.print_statistics()
        else:
            print("Optimal solution found")

        self.opt_xi, self.opt_ui = self.get_solution(solver, iteration)

        return self.opt_xi, self.opt_ui, status

    def generate_init_guess(self, x0):

        if self.model_type.casefold() == 'kinematic bicycle'.casefold():
            u1 = np.zeros(self.N)  # Steering angle rate
            u2 = 0.5*np.ones(self.N)  # acceleration
            u3 = 0.5*np.ones(self.N)  # Virtual acceleration/ Projected acceleration

            self.init_u = np.column_stack((u1, u2, u3))

            self.init_x = np.array([x0])
            x = x0
            for u in self.init_u:
                x = self.ocp.system.simulate(x, u)
                self.init_x = np.vstack((self.init_x, x.T))
        else:
            u1 = np.zeros(self.N)  # Steering angle rate
            u2 = 0.1 * np.ones(self.N)  # acceleration
            u3 = 0.1 * np.ones(self.N)  # Virtual acceleration/ Projected acceleration
            self.init_u = np.column_stack((u1, u2, u3))
            self.init_x = np.array([x0])
            x = x0
            for u in self.init_u:
                x = self.ocp.system.simulate(x, u)
                self.init_x = np.vstack((self.init_x, x.T))

        return self.init_x, self.init_u

    def warm_start(self):

        xf = self.opt_xi[-1, :]
        uf = self.opt_ui[-1, :]

        xN_1 = self.ocp.system.simulate(xf, uf).reshape(1, xf.shape[0])

        self.init_x = np.vstack((self.opt_xi[1:, :], xN_1))
        self.init_u = np.vstack((self.opt_ui[1:, :], uf))

        return self.init_x, self.init_u


class TrackingMPC:

    def __init__(self, model_type, horizon, shooting_nodes, track, x0, Q, R, constrained=True):

        self.ocp = OCP(horizon=horizon, shooting_nodes=shooting_nodes, track_length=track.length,
                       track_width=track.width, x0=x0, model_type=model_type, constrained=constrained)

        if model_type.casefold() == 'Kinematic Bicycle'.casefold():
            self.ocp_solver = self.ocp.build_tracking_OCP(Q=Q, R=R)
        else:
            self.ocp_solver = None

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

        self.t_samp = self.ocp.t_samp


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


# class CurvatureMPC:
#     def __init__(self, model_type, horizon, shooting_nodes, track_length, track_width, x0, weights, n_params,
#                  constrained=True):
#
#         self.ocp = OCP(horizon=horizon, shooting_nodes=shooting_nodes, track_length=track_length,
#                        track_width=track_width, x0=x0, weights=weights, model_type=model_type, constrained=constrained,
#                        R=None, ocp_type='CurvatureMPC')
#
#         self.ocp_solver = self.ocp.build_curvature_OCP(n_params=n_params)
#
#         self.nx = self.ocp.system.nx
#         self.nu = self.ocp.system.nu
#         self.N = shooting_nodes
#         self.model_type = model_type
#
#         self.opt_xi = None
#         self.opt_ui = None
#
#         self.opt_x = np.array([])
#         self.opt_u = np.array([])
#
#         self.init_x = None
#         self.init_u = None
#
#         self.t_samp = self.ocp.t_samp




    # def build_curvature_OCP(self, n_params=1):
    #
    #     ocp = AcadosOcp()
    #     model = self.system.spatial_model()
    #
    #     # Create ACADOS Model
    #     model_ac = AcadosModel()
    #     model_ac.name = model.name
    #     model_ac.f_impl_expr = model.f_impl_expr
    #     model_ac.f_expl_expr = model.f_expl_expr
    #     model_ac.x = model.x
    #     model_ac.xdot = model.xdot
    #     model_ac.u = model.u
    #     ocp.model = model_ac
    #
    #     """ SETTING PARAMETERS """
    #     # Initialize parameter values
    #     model_ac.p = model.p
    #     ocp.parameter_values = np.zeros(n_params)
    #
    #     """ COST FUNCTION """
    #     ocp.cost.cost_type = 'EXTERNAL'
    #     ocp.cost.cost_type_e = 'EXTERNAL'
    #
    #     e_y_k = model_ac.x[0]
    #     e_psi_k = model_ac.x[1]
    #     v_k = model_ac.u[0]
    #     rho_s_k = model_ac.p[0]
    #
    #     model.cost_expr = - (v_k * rho_s_k * cos(e_psi_k))/(rho_s_k - e_y_k)
    #     model.cost_expr_e = 0  # - self.qtheta*v_k*self.t_samp
    #
    #     ocp.model.cost_expr_ext_cost = model.cost_expr
    #     ocp.model.cost_expr_ext_cost_e = model.cost_expr_e
    #
    #     """ STATE CONSTRAINTS """
    #     # Constraint at first shooting node
    #     ocp.constraints.x0 = self.x0
    #
    #     if self.constrained:
    #         constraint = types.SimpleNamespace()
    #
    #         dmax = self.track_width/2
    #         constraint.expr = e_y_k
    #         ocp.model.con_h_expr_0 = constraint.expr
    #         ocp.constraints.lh_0 = np.array([-dmax])
    #         ocp.constraints.uh_0 = np.array([dmax])
    #
    #         ocp.model.con_h_expr = constraint.expr
    #         ocp.constraints.lh = np.array([-dmax])
    #         ocp.constraints.uh = np.array([dmax])
    #
    #         ocp.model.con_h_expr_e = constraint.expr
    #         ocp.constraints.lh_e = np.array([-dmax])
    #         ocp.constraints.uh_e = np.array([dmax])
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
    #     ocp.solver_options.tol = 1e-1
    #     ocp.solver_options.regularize_method = 'MIRROR'
    #
    #     ocp.code_export_directory = 'CurveMPCC_OCP'
    #     solver = AcadosOcpSolver(ocp, json_file=f"{self.ocp_type}_{self.model_type}_OCP.json")
    #
    #     return solver