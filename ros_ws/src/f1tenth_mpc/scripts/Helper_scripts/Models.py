from casadi import *
import types


class CarModel:
    def __init__(self, h=1/20, model_typ='Kinematic Bicycle', initial=False):

        self.model_type = model_typ  # Model type
        self.nx = None  # Initializing number of states
        self.nu = None  # Initializing number of inputs
        self.initial = initial

        self.h = h  # Discretisation rate
        self.model = None

        self.aug = None

        # Model Parameters
        self.p = types.SimpleNamespace()
        self.p.lr = 0.17145  # m (Distance from CoG to center of rear axle)
        self.p.lf = 0.15875  # m (Distance from CoG to center of front axle)
        self.p.Csr = 5.4562  # Cornering stiffness rear wheel
        self.p.Csf = 4.718  # Cornering stiffness front wheel
        self.p.g = 9.8  # m/s² (Acceleration due to gravity)
        self.p.hcg = 0.074  # m (Height to center of gravity)
        self.p.U = 0.523  # Friction coefficient
        self.p.m = 3.47  # Mass of car
        self.p.Iz = 0.04712  # kg m² Moment of Inertia
        self.p.lwb = (self.p.lf + self.p.lr)

        # Tyre dynamics parameters
        self.p.Kv = 0.1
        self.p.B = 10
        self.p.C = 1.9
        self.p.D = 1

        # max_speed: 7.  # meters/second
        # max_steering_angle: 0.4189  # radians
        # max_accel: 7.51  # meters/second^2
        # max_decel: 8.26  # meters/second^2
        # max_steering_vel: 3.2  # radians/second
        # friction_coeff: 0.523  # - (complete estimate)
        # height_cg: 0.074  # m (roughly measured to be 3.25 in)
        # l_cg2rear: 0.17145  # m (decently measured to be 6.75 in)
        # l_cg2front: 0.15875  # m (decently measured to be 6.25 in)
        # C_S_front: 4.718  # .79 # 1/rad ? (estimated weight/4)
        # C_S_rear: 5.4562  # .79 # 1/rad ? (estimated weight/4)
        # mass: 3.47  # kg (measured on car 'lidart')
        # moment_inertia: .04712  # kg m^2

    def MPCC_kinematic_model(self):

        # STATES
        x1 = SX.sym('X')  # x-coordinate of Center of Gravity of car
        x2 = SX.sym('Y')  # y-coordinate of Center of Gravity of car
        x3 = SX.sym('psi')  # Orientation of car
        x4 = SX.sym('s')  # Measure of progress along track
        x5 = SX.sym('delta')  # Steering angle
        x6 = SX.sym('v')  # Velocity
        x7 = SX.sym('v_s')  # projected velocity

        # INPUTS
        u1 = SX.sym('v_delta')  # Steering angle rate
        u2 = SX.sym('a')  # acceleration
        u3 = SX.sym('a_s')  # Projected acceleration

        # DIFFERENTIAL STATES
        x1_dot = SX.sym('X_dot')
        x2_dot = SX.sym('Y_dot')
        x3_dot = SX.sym('psi_dot')
        x4_dot = SX.sym('s_dot')
        x5_dot = SX.sym('delta_dot')
        x6_dot = SX.sym('v_dot')
        x7_dot = SX.sym('v_s_dot')

        model_name = "kinematic_model"
        xdot = vertcat(x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot)

        # STATE EQUATIONS
        dx1 = x6 * cos(x3)
        dx2 = x6 * sin(x3)
        dx3 = (x6 * tan(x5)) / (self.p.lr + self.p.lf)
        dx4 = x7
        dx5 = u1
        dx6 = u2
        dx7 = u3

        f_expl = vertcat(dx1, dx2, dx3, dx4, dx5, dx6, dx7)
        f_impl = xdot - f_expl

        self.model = types.SimpleNamespace()
        self.model.name = model_name
        self.model.x = vertcat(x1, x2, x3, x4, x5, x6, x7)
        self.model.u = vertcat(u1, u2, u3)
        self.model.f_impl_expr = f_impl
        self.model.f_expl_expr = f_expl

        self.model.xdot = xdot
        self.nx = self.model.x.size()[0]
        self.nu = self.model.u.size()[0]

        # Model Constraints
        self.model.delta_min = -0.4  # -0.4891
        self.model.delta_max = 0.4  # 0.4891
        self.model.delta_idx = 4

        self.model.v_min = -1.5
        self.model.v_max = 4.5
        self.model.v_idx = 5

        self.model.v_s_min = 0.0
        self.model.v_s_max = 4.15
        self.model.v_s_idx = 6

        self.model.v_delta_min = -3.2
        self.model.v_delta_max = 3.2

        self.model.a_min = -7.5
        self.model.a_max = 7.5

        self.model.a_s_min = 0
        self.model.a_s_max = 7.50  # 7.51

        self.model.xmin = np.array([self.model.delta_min, self.model.v_min, self.model.v_s_min])
        self.model.xmax = np.array([self.model.delta_max, self.model.v_max, self.model.v_s_max])
        self.model.x_idx = np.array([self.model.delta_idx, self.model.v_idx, self.model.v_s_idx])

        self.model.umin = np.array([self.model.v_delta_min, self.model.a_min, self.model.a_s_min])
        self.model.umax = np.array([self.model.v_delta_max, self.model.a_max, self.model.a_s_max])

        return self.model

    def MPCC_dynamic_model(self):

        # STATES
        x1 = SX.sym('X')  # X coordinate of CoG
        x2 = SX.sym('Y')  # Y coordinate of CoG
        x3 = SX.sym('psi')  # Orientation of car
        x4 = SX.sym('vx')  # Longitudinal velocity of CoG
        x5 = SX.sym('vy')  # Lateral velocity of CoG
        x6 = SX.sym('w')  # Angular velocity of car
        x7 = SX.sym('s')  # Progress along the reference

        # Adding inputs as states to control the rate of change of inputs
        x8 = SX.sym('delta')  # Steering angle
        x9 = SX.sym('v_x_ref')  # Desired longitudinal velocity
        x10 = SX.sym('v_s')  # Rate of progress along the reference

        # INPUTS
        u1 = SX.sym('v_delta')  # Steering angle rate
        u2 = SX.sym('a_x_ref')  # Longitudinal acceleration
        u3 = SX.sym('a_s')  # Acceleration of progress along the reference

        # DIFFERENTIAL STATES
        x1_dot = SX.sym('X_dot')
        x2_dot = SX.sym('Y_dot')
        x3_dot = SX.sym('psi_dot')
        x4_dot = SX.sym('vx_dot')
        x5_dot = SX.sym('vy_dot')
        x6_dot = SX.sym('w_dot')
        x7_dot = SX.sym('s_dot')

        x8_dot = SX.sym('delta_dot')
        x9_dot = SX.sym('v_x_ref_dot')
        x10_dot = SX.sym('v_s_dot')

        model_name = "Dynamic_Pacejka_model"
        xdot = vertcat(x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot, x8_dot, x9_dot, x10_dot)

        # STATE EQUATIONS
        eps = 1e-12
        alpha_r = atan((self.p.lr * x6 - x5) / (x4 + eps)) + x8
        alpha_f = -atan((self.p.lf * x6 + x5) / (x4 + eps))

        F_rx = (x9 - x4) * self.p.Kv
        F_ry = self.p.D * sin(self.p.C * atan(self.p.B * alpha_r))
        F_fy = self.p.D * sin(self.p.C * atan(self.p.B * alpha_f))

        dx1 = x4 * cos(x3) - x5 * sin(x3)
        dx2 = x4 * sin(x3) + x5 * cos(x3)
        dx3 = x6
        dx4 = (1/self.p.m) * (F_rx - F_fy * sin(x8) + self.p.m * x5 * x6)
        dx5 = (1/self.p.m) * (F_ry - F_fy * cos(x8) - self.p.m * x5 * x6)
        dx6 = (1/self.p.Iz) * (self.p.lf * F_fy * cos(x8) - self.p.lr * F_ry)
        dx7 = x10
        dx8 = u1
        dx9 = u2
        dx10 = u3

        f_expl = vertcat(dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9, dx10)
        f_impl = xdot - f_expl

        self.model = types.SimpleNamespace()
        self.model.name = model_name
        self.model.x = vertcat(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
        self.model.u = vertcat(u1, u2, u3)
        self.model.f_impl_expr = f_impl
        self.model.f_expl_expr = f_expl

        self.model.xdot = xdot
        self.nx = self.model.x.size()[0]
        self.nu = self.model.u.size()[0]

        # Model Constraints
        self.model.delta_min = -0.34  # -0.4891
        self.model.delta_max = 0.34  # 0.4891
        self.model.delta_idx = 6

        self.model.vx_min = 0.00
        self.model.vx_max = 4.5
        self.model.vx_idx = 3

        self.model.v_s_min = 0.0
        self.model.v_s_max = 4.25
        self.model.v_s_idx = 9

        self.model.v_delta_min = -3.2
        self.model.v_delta_max = 3.2

        self.model.a_min = -7.00
        self.model.a_max = 7.00

        self.model.a_s_min = 0
        self.model.a_s_max = 5.0  # 7.51

        self.model.xmin = np.array([self.model.delta_min, self.model.vx_min, self.model.v_s_min])
        self.model.xmax = np.array([self.model.delta_max, self.model.vx_max, self.model.v_s_max])
        self.model.x_idx = np.array([self.model.delta_idx, self.model.vx_idx, self.model.v_s_idx])

        self.model.umin = np.array([self.model.v_delta_min, self.model.a_min, self.model.a_s_min])
        self.model.umax = np.array([self.model.v_delta_max, self.model.a_max, self.model.a_s_max])

        return self.model

    def TrackingMPC_kinematic_model(self):
        x1 = SX.sym('X')  # x-coordinate of Center of Gravity of car
        x2 = SX.sym('Y')  # y-coordinate of Center of Gravity of car
        x3 = SX.sym('psi')  # Orientation of car
        x4 = SX.sym('delta')  # Steering angle
        x5 = SX.sym('v')  # Velocity

        # INPUTS
        u1 = SX.sym('v_delta')  # Steering angle rate
        u2 = SX.sym('a')  # acceleration

        # DIFFERENTIAL STATES
        x1_dot = SX.sym('X_dot')
        x2_dot = SX.sym('Y_dot')
        x3_dot = SX.sym('psi_dot')
        x4_dot = SX.sym('delta_dot')
        x5_dot = SX.sym('v_dot')

        model_name = "kinematic_model"
        xdot = vertcat(x1_dot, x2_dot, x3_dot, x4_dot, x5_dot)

        # STATE EQUATIONS
        dx1 = x5 * cos(x3)
        dx2 = x5 * sin(x3)
        dx3 = (x5 * tan(x4)) / (self.p.lr + self.p.lf)
        dx4 = u1
        dx5 = u2

        f_expl = vertcat(dx1, dx2, dx3, dx4, dx5)
        f_impl = xdot - f_expl

        self.model = types.SimpleNamespace()
        self.model.name = model_name
        self.model.x = vertcat(x1, x2, x3, x4, x5)
        self.model.u = vertcat(u1, u2)
        self.model.f_impl_expr = f_impl
        self.model.f_expl_expr = f_expl

        self.model.xdot = xdot
        self.nx = self.model.x.size()[0]
        self.nu = self.model.u.size()[0]

        # Model Constraints
        self.model.delta_min = -0.3  # -0.4891
        self.model.delta_max = 0.3  # 0.4891
        self.model.delta_idx = 4

        self.model.v_min = -2.00
        self.model.v_max = 4.75
        self.model.v_idx = 5

        self.model.v_delta_min = -3.2
        self.model.v_delta_max = 3.2

        self.model.a_min = -7.51
        self.model.a_max = 7.51

        self.model.xmin = np.array([self.model.delta_min, self.model.v_min])
        self.model.xmax = np.array([self.model.delta_max, self.model.v_max])
        self.model.x_idx = np.array([self.model.delta_idx, self.model.v_idx])

        self.model.umin = np.array([self.model.v_delta_min, self.model.a_min])
        self.model.umax = np.array([self.model.v_delta_max, self.model.a_max])

        return self.model

    def f(self, x, u):
        sys_fcn = Function('System_Function', [self.model.x, self.model.u], [self.model.f_expl_expr])
        xdot = sys_fcn(x, u)
        return xdot

    def simulate(self, x, u):
        k1 = self.f(x, u)
        k2 = self.f(x + self.h / 2 * k1, u)
        k3 = self.f(x + self.h / 2 * k2, u)
        k4 = self.f(x + self.h * k3, u)
        xnext = x + self.h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xnext = xnext.full()
        return xnext

# def measure_states(car_odometry, theta, v_theta, t_samp, p=None, delta=0.0, model="Bicycle"):
#
#     if model.casefold() == "Bicycle".casefold():
#
#         pose = car_odometry.pose
#         steer_ang = delta
#         vel = car_odometry.linear_vel[0]
#         w = car_odometry.angular_vel[2]
#         beta = arctan2(tan(steer_ang)*p.lr, (p.lr+p.lf))
#         theta += v_theta*t_samp
#         x = np.array((pose[0], pose[1], delta, vel, pose[2], w, beta, theta))
#
#     elif model.casefold() == "Dynamic".casefold():
#
#         pose = car_odometry.pose
#         vx = car_odometry.linear_vel[0]
#         vy = car_odometry.linear_vel[1]
#         w = car_odometry.angular_vel[2]
#         theta += v_theta * t_samp
#
#         x = np.array((pose[0], pose[1], pose[2], vx, vy, w, theta))
#     else:
#         print("Please provide valid model code")
#         x = None
#
#     return x

# def KinematicBicycleModel(start_pose, lr=0.17145, lf=0.15875, model_name="Kinematic_Bicycle_Model"):
#     """
#
#         Function referred from acados/examples/getting_started/pendulum_model.py
#         :arguments:
#             lr : Distance from center of gravity of car to the rear axle
#             lf : Distance from center of gravity of car to the front axle
#
#         :return:
#             System Model: AcadosModel() object
#     """
#     # STATES
#     X = SX.sym('X')  # x-coordinate of Center of Gravity of car
#     Y = SX.sym('Y')  # y-coordinate of Center of Gravity of car
#     psi = SX.sym('psi')  # Orientation of car
#     theta = SX.sym('theta')  # Arc length
#
#     # INPUTS
#     delta = SX.sym('delta')  # Steering angle
#     v = SX.sym('v')  # Actual velocity of car
#     v_theta = SX.sym('v_theta')  # Projected velocity of car
#
#     x = vertcat(X, Y, psi, theta)  # State vector
#     u = vertcat(delta, v, v_theta)  # Input vector
#
#     # DIFFERENTIAL STATES
#     X_dot = SX.sym('sx_dot')
#     Y_dot = SX.sym('sy_dot')
#     psi_dot = SX.sym('psi_dot')
#     theta_dot = SX.sym('theta_dot')
#
#     xdot = vertcat(X_dot, Y_dot, psi_dot, theta_dot)
#
#     # EXPLICIT FORM OF STATE EQUATION
#     f_expl = vertcat(
#                 u[1]*cos(x[2]),
#                       u[1]*sin(x[2]),
#                      (u[1]*tan(u[0]))/(lr+lf),
#                       u[2]
#                      )
#
#     f_impl = xdot - f_expl
#
#     model = types.SimpleNamespace()
#     model.name = model_name
#
#     model.f_impl_expr = f_impl
#     model.f_expl_expr = f_expl
#
#     X_ref = SX.sym('X_ref')
#     Y_ref = SX.sym('Y_ref')
#     psi_ref = SX.sym('psi_ref')
#     model.p = vertcat(X_ref, Y_ref, psi_ref)
#
#     model.x = x
#     model.x0 = start_pose
#     model.xdot = xdot
#
#     model.u = u
#
#     return model


# def bicycle_model_const_vel(lr=0.17145, lf=0.15875, model_name="Kinematic_Bicycle_Model"):
#     """
#
#         Function referred from acados/examples/getting_started/pendulum_model.py
#         :arguments:
#             lr : Distance from center of gravity of car to the rear axle
#             lf : Distance from center of gravity of car to the front axle
#
#         :return:
#             System Model: AcadosModel() object
#     """
#     # STATES
#     sx = SX.sym('sx')  # x-coordinate of Center of Gravity of car
#     sy = SX.sym('sy')  # y-coordinate of Center of Gravity of car
#     psi = SX.sym('psi')  # Orientation of car
#
#     # INPUTS
#     delta = SX.sym('delta')  # Heading angle
#     v = 2.0
#
#     x = vertcat(sx, sy, psi)  # Vector of states
#     u = delta
#
#     # DIFFERENTIAL STATES
#     sx_dot = SX.sym('sx_dot')
#     sy_dot = SX.sym('sy_dot')
#     psi_dot = SX.sym('psi_dot')
#
#     xdot = vertcat(sx_dot, sy_dot, psi_dot)
#
#     # EXPLICIT FORM OF STATE EQUATION
#     f_expl = vertcat(
#                 v*cos(x[2]),
#                       v*sin(x[2]),
#                      (v*tan(u[0]))/(lr+lf)
#                      )
#
#     f_impl = xdot - f_expl
#
#     model = types.SimpleNamespace()
#     model.name = model_name
#
#     model.f_impl_expr = f_impl
#     model.f_expl_expr = f_expl
#
#     model.x = x
#     model.x0 = np.array([0.0, 0.0, 0.0])
#     model.xdot = xdot
#
#     model.u = u
#
#     return model

# class SimulatedCar(CarModel):
#     def __init__(self, g, l, b, m, h, x0):
#         super(SimulatedInvertedPendulum, self).__init__(g, l, b, m, h)
#
#         self.x = x0.reshape(self.nx, 1)
#
#     def step(self, u):
#         """
#     Function for finding the next state using the Runge-Kutta 4 discretisation scheme
#     Discrete time dynamics
#
#     input argument:
#     a. u - current input
#
#     output:
#     a. x - next state
#         """
#         k1 = self.system(self.x, u)
#         k2 = self.system(self.x + self.h / 2 * k1, u)
#         k3 = self.system(self.x + self.h / 2 * k2, u)
#         k4 = self.system(self.x + self.h * k3, u)
#         self.x = self.x + self.h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
#         self.x = self.x.full()
#
#         return self.x
#
#     def measure(self):
#         return self.x
#
#     def apply(self, u):
#
#         k1 = self.system(self.x, u)
#         k2 = self.system(self.x + self.h / 2 * k1, u)
#         k3 = self.system(self.x + self.h / 2 * k2, u)
#         k4 = self.system(self.x + self.h * k3, u)
#         self.x = self.x + self.h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
#         self.x = self.x.full()
#
#
# class RealInvertedPendulum(AbstractInvertedPendulum):
#     def __init__(self, g, l, b, m, h, port, baud_rate):
#         super(RealInvertedPendulum, self).__init__(g, l, b, m, h)
#
#         self.ser = SerialCom(port, baud_rate)
#         self.x = self.ser.measure()
#
#     def step(self, u):
#         self.ser.apply(u)
#         time.sleep(self.h)
#         self.x = self.ser.measure()
#
#         return self.x
#
#     def apply(self, u):
#         self.ser.apply(u)
#
#     def measure(self):
#         # dig the latest state value from the buffer
#         return self.ser.measure()

#     def Bicycle_Model(self, track_length):
#
#         self.model = types.SimpleNamespace()
#
#         # STATES
#         x1 = SX.sym('X')  # x-coordinate of Center of Gravity of car
#         x2 = SX.sym('Y')  # y-coordinate of Center of Gravity of car
#         x3 = SX.sym('delta')  # Steering angle
#         x4 = SX.sym('v')  # Velocity
#         x5 = SX.sym('psi')  # Orientation of car
#         x6 = SX.sym('w')  # Angular velocity
#         x7 = SX.sym('beta')  # Slip angle
#         x8 = SX.sym('theta')  # Progress along the reference
#
#         # INPUTS
#         u1 = SX.sym('v_delta')  # Steering angle rate
#         u2 = SX.sym('a_long')  # Longitudinal acceleration
#         u3 = SX.sym('v_theta')  # Rate of progress along the reference
#
#         # DIFFERENTIAL STATES
#         x1_dot = SX.sym('X_dot')
#         x2_dot = SX.sym('Y_dot')
#         x3_dot = SX.sym('delta_dot')
#         x4_dot = SX.sym('v_dot')
#         x5_dot = SX.sym('psi_dot')
#         x6_dot = SX.sym('w_dot')
#         x7_dot = SX.sym('beta_dot')
#         x8_dot = SX.sym('theta_dot')
#
#         xdot = vertcat(x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot, x8_dot)
#
#         # EQUATIONS
#         dx1 = x4 * np.cos(x5 + x7)  # X_dot
#         dx2 = x4 * np.sin(x5 + x7)  # Y_dot
#         dx3 = u1  # delta_dot
#         dx4 = u2  # v_dot
#
#         dx8 = u3  # theta_dot
#
#         p = self.p
#         if not self.initial:
#             dx5 = x6  # psi_dot
#
#             a1 = p.lf * p.Csf * (p.g * p.lr - u2 * p.hcg)
#             a2 = p.lr * p.Csr * (p.g * p.lf + u2 * p.hcg) - p.lf * p.Csf * (p.g * p.lr - u2 * p.hcg)
#             a3 = (p.lf ** 2) * p.Csf * (p.g * p.lr - u2 * p.hcg) + (p.lr ** 2) * p.Csr * (p.g * p.lf + u2 * p.hcg)
#             dx6 = (p.U * p.m) / (p.Iz * (p.lr + p.lf)) * (a1 * x3 + a2 * x7 - a3 * (x6 / x4))  # w_dot
#
#             b1 = p.Csf * (p.g * p.lr - u2 * p.hcg)
#             b2 = p.Csr * (p.g * p.lf + u2 * p.hcg) + p.Csf * (p.g * p.lr - u2 * p.hcg)
#             b3 = p.Csr * (p.g * p.lf + u2 * p.hcg) * p.lr - p.Csf * (p.g * p.lr - u2 * p.hcg)
#             dx7 = p.U / (x4 * (p.lr + p.lf)) * (b1 * x3 - b2 * x7 + b3 * (x6 / x4)) - x6  # beta_dot
#
#             self.model.name = "Bicycle_Model"
#
#         else:
#             dx5 = (x4 * cos(x7) * tan(x3)) / p.lwb
#             dx6 = (u2 * tan(x3) + (x4 * u1) / (cos(x3) ** 2)) / p.lwb
#             dx7 = (p.lr * p.lwb * u1) / ((p.lwb * cos(x3)) ** 2 + (p.lr * sin(x3)) ** 2)
#
#             self.model.name = "Non_singular_Bicycle_Model"
#
#         f_expl = vertcat(dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8)
#         f_impl = xdot - f_expl
#
#         self.model.p = p
#         self.model.x = vertcat(x1, x2, x3, x4, x5, x6, x7, x8)
#         self.model.u = vertcat(u1, u2, u3)
#         self.model.xdot = vertcat(x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot, x8_dot)
#         self.model.f_impl_expr = f_impl
#         self.model.f_expl_expr = f_expl
#         self.model.x0 = self.init_state
#
#         self.nx = self.model.x.size()[0]
#         self.nu = self.model.u.size()[0]
#
#         # Model constraints
#         self.model.steer_ang_vel_min = 0
#         self.model.steer_ang_vel_max = 3.2
#         self.model.a_long_min = -7.51
#         self.model.a_long_max = 7.51
#         self.model.v_theta_min = 0
#         self.model.v_theta_max = 3
#
#         self.model.delta_min = -0.4189
#         self.model.delta_max = 0.4189
#         self.model.delta_idx = 2
#
#         self.model.v_min = 0
#         self.model.v_max = 7
#         self.model.v_idx = 3
#
#         self.model.psi_min = -np.pi
#         self.model.psi_max = np.pi
#         self.model.psi_idx = 4
#
#         self.model.theta_min = 0
#         self.model.theta_max = track_length
#         self.model.theta_idx = self.model.x.size()[0] - 1
#
#         self.model.umin = np.array([self.model.steer_ang_vel_min, self.model.a_long_min, self.model.v_theta_min])
#         self.model.umax = np.array([self.model.steer_ang_vel_max, self.model.a_long_max, self.model.v_theta_max])
#
#         self.model.xmin = np.array([self.model.delta_min, self.model.v_min, self.model.psi_min, self.model.theta_min])
#         self.model.xmax = np.array([self.model.delta_max, self.model.v_max, self.model.psi_max, self.model.theta_max])
#         self.model.x_idx = np.array([self.model.delta_idx, self.model.v_idx, self.model.psi_idx, self.model.theta_idx])
#
#         return self.model
#
#     def Dynamic_model(self, track_length):
#
#         # STATES
#         x1 = SX.sym('X')  # x-coordinate of Center of Gravity of car
#         x2 = SX.sym('Y')  # y-coordinate of Center of Gravity of car
#         x3 = SX.sym('psi')  # Orientation of car
#         x4 = SX.sym('vx')  # Velocity x
#         x5 = SX.sym('vy')  # Velocity y
#         x6 = SX.sym('w')  # Angular velocity/ yaw rate
#         x7 = SX.sym('theta')  # Progress measurement along track (Augmented state)
#
#         # INPUTS
#         u1 = SX.sym('vx_ref')  # Longitudinal velocity
#         u2 = SX.sym('delta')  # Steering angle
#         u3 = SX.sym('v_theta')  # Rate of progress along track
#
#         # DIFFERENTIAL STATES
#         x1_dot = SX.sym('X_dot')
#         x2_dot = SX.sym('Y_dot')
#         x3_dot = SX.sym('psi_dot')
#         x4_dot = SX.sym('vx_dot')
#         x5_dot = SX.sym('vy_dot')
#         x6_dot = SX.sym('w_dot')
#         x7_dot = SX.sym('theta_dot')
#
#         xdot = vertcat(x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot)
#
#         p = self.p
#         # EQUATIONS
#         dx1 = x4 * cos(x3) - x5 * sin(x3)
#         dx2 = x4 * sin(x3) + x5 * cos(x3)
#         dx3 = x6
#         dx4 = (u1 - x4) * p.Kv
#
#         alphaf = -atan((x6 * p.lf + x5) / x4) + u2
#         alphar = atan((x6 * p.lr - x5) / x4)
#
#         Fry = p.D * sin(p.C * atan(p.B * alphar))
#         Ffy = p.D * sin(p.C * atan(p.B * alphaf))
#
#         dx5 = (Fry + Ffy * cos(u2) - p.m * x4 * x6) / p.m
#         dx6 = (Ffy * p.lf * cos(u2) - Fry * p.lr) / p.Iz
#         dx7 = u3
#
#         f_expl = vertcat(dx1, dx2, dx3, dx4, dx5, dx6, dx7)
#         f_impl = xdot - f_expl
#
#         self.model = types.SimpleNamespace()
#         self.model.name = "Dynamic_Pacjeka_Model"
#         self.model.x = vertcat(x1, x2, x3, x4, x5, x6, x7)
#         self.model.u = vertcat(u1, u2, u3)
#         self.model.xdot = vertcat(x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot)
#         self.model.f_impl_expr = f_impl
#         self.model.f_expl_expr = f_expl
#         self.model.x0 = self.init_state
#         self.model.p = p
#
#         self.nx = self.model.x.size()[0]
#         self.nu = self.model.u.size()[0]
#
#         # Model constraints
#         self.model.delta_min = -0.4189
#         self.model.delta_max = 0.4189
#
#         self.model.v_min = -2
#         self.model.v_max = 5
#
#         self.model.v_theta_min = 0
#         self.model.v_theta_max = 3
#
#         self.model.psi_min = -np.pi
#         self.model.psi_max = np.pi
#         self.model.psi_idx = 2
#
#         self.model.theta_min = 0
#         self.model.theta_max = track_length
#         self.model.theta_idx = self.nx - 1
#
#         self.model.umin = np.array([self.model.v_min, self.model.delta_min, self.model.v_theta_min])
#         self.model.umax = np.array([self.model.v_max, self.model.delta_max, self.model.v_theta_max])
#
#         self.model.xmin = np.array([self.model.psi_min, self.model.theta_min])
#         self.model.xmax = np.array([self.model.psi_max, self.model.theta_max])
#         self.model.x_idx = np.array([self.model.psi_idx, self.model.theta_idx])
#
#         return self.model


    # def kinematic_model(self):
    #
    #     self.model = types.SimpleNamespace()
    #
    #     x1 = SX.sym('X')  # x-coordinate of Center of Gravity of car
    #     x2 = SX.sym('Y')  # y-coordinate of Center of Gravity of car
    #     x3 = SX.sym('psi')  # Orientation of car
    #
    #     # INPUTS
    #     u1 = SX.sym('delta')  # Steering angle
    #     u2 = SX.sym('v')  # Velocity of car
    #
    #     # DIFFERENTIAL STATES
    #     x1_dot = SX.sym('X_dot')
    #     x2_dot = SX.sym('Y_dot')
    #     x3_dot = SX.sym('psi_dot')
    #
    #     model_name = "Kinematic_Model"
    #     xdot = vertcat(x1_dot, x2_dot, x3_dot)
    #
    #     # STATE EQUATIONS
    #     dx1 = u2 * cos(x3)
    #     dx2 = u2 * sin(x3)
    #     dx3 = (u2 * tan(u1)) / (self.p.lr + self.p.lf)
    #
    #     f_expl = vertcat(dx1, dx2, dx3)
    #
    #     f_impl = xdot - f_expl
    #
    #     self.model.name = model_name
    #     self.model.x = vertcat(x1, x2, x3)
    #     self.model.u = vertcat(u1, u2)
    #     self.model.f_impl_expr = f_impl
    #     self.model.f_expl_expr = f_expl
    #
    #     self.model.x0 = self.init_state
    #     self.model.xdot = xdot
    #     self.nx = self.model.x.size()[0]
    #     self.nu = self.model.u.size()[0]
    #
    #     return self.model


   # def kinematic_bicycle_model(self, track_length, augmented=True):
   #
   #      self.model = types.SimpleNamespace()
   #
   #      # STATES
   #      x1 = SX.sym('X')  # x-coordinate of Center of Gravity of car
   #      x2 = SX.sym('Y')  # y-coordinate of Center of Gravity of car
   #      x3 = SX.sym('psi')  # Orientation of car
   #
   #      # INPUTS
   #      u1 = SX.sym('delta')  # Steering angle
   #      u2 = SX.sym('v')  # Velocity of car
   #
   #      # DIFFERENTIAL STATES
   #      x1_dot = SX.sym('X_dot')
   #      x2_dot = SX.sym('Y_dot')
   #      x3_dot = SX.sym('psi_dot')
   #
   #      self.aug = augmented
   #      if not augmented:
   #          model_name = "Kinematic_Bicycle_Model"
   #          xdot = vertcat(x1_dot, x2_dot, x3_dot)
   #
   #          # STATE EQUATIONS
   #          dx1 = u2 * cos(x3)
   #          dx2 = u2 * sin(x3)
   #          dx3 = (u2 * tan(u1)) / (self.p.lr + self.p.lf)
   #
   #          f_expl = vertcat(dx1, dx2, dx3)
   #
   #          f_impl = xdot - f_expl
   #
   #          self.model.name = model_name
   #          self.model.x = vertcat(x1, x2, x3)
   #          self.model.u = vertcat(u1, u2)
   #
   #      else:
   #          model_name = "Kinematic_Bicycle_Model_augmented"
   #          # ADDITIONAL STATE
   #          x4 = SX.sym('theta')  # Arc length
   #          # ADDITIONAL INPUT
   #          u3 = SX.sym('v_theta')  # Projected velocity of car
   #          # ADDITIONAL DIFFERENTIAL STATE
   #          x4_dot = SX.sym('theta_dot')
   #
   #          xdot = vertcat(x1_dot, x2_dot, x3_dot, x4_dot)
   #
   #          # STATE EQUATIONS
   #          dx1 = u2 * cos(x3)
   #          dx2 = u2 * sin(x3)
   #          dx3 = (u2 * tan(u1)) / (self.p.lr + self.p.lf)
   #          dx4 = u3
   #
   #          f_expl = vertcat(dx1, dx2, dx3, dx4)
   #          f_impl = xdot - f_expl
   #          self.model.name = model_name
   #          self.model.x = vertcat(x1, x2, x3, x4)
   #          self.model.u = vertcat(u1, u2, u3)
   #
   #      self.model.f_impl_expr = f_impl
   #      self.model.f_expl_expr = f_expl
   #
   #      self.model.x0 = self.init_state
   #      self.model.xdot = xdot
   #      self.nx = self.model.x.size()[0]
   #      self.nu = self.model.u.size()[0]
   #
   #      # Model Constraints
   #      self.model.v_min = -2.0
   #      self.model.v_max = 4.75
   #
   #      self.model.delta_min = -0.4891
   #      self.model.delta_max = 0.4891
   #
   #      if augmented:
   #          self.model.v_theta_min = 0.0
   #          self.model.v_theta_max = 4.5
   #
   #          self.model.theta_min = 0
   #          self.model.theta_max = 1000*track_length
   #          self.model.theta_idx = self.nx - 1
   #
   #      self.model.umin = np.array([self.model.delta_min, self.model.v_min, self.model.v_theta_min])
   #      self.model.umax = np.array([self.model.delta_max, self.model.v_max, self.model.v_theta_max])
   #
   #      self.model.xmin = np.array([self.model.theta_min])
   #      self.model.xmax = np.array([self.model.theta_max])
   #      self.model.x_idx = np.array([self.model.theta_idx])
   #
   #      return self.model

  # def spatial_model(self):
  #
  #       # States
  #       x1 = SX.sym('e_y')
  #       x2 = SX.sym('e_psi')
  #
  #       # Inputs
  #       u1 = SX.sym('v')
  #       u2 = SX.sym('delta')
  #
  #       # DIFFERENTIAL STATES
  #       x1_prime = SX.sym('e_y_prime')
  #       x2_prime = SX.sym('e_psi_prime')
  #
  #       model_name = "spatial_model"
  #       xdot = vertcat(x1_prime, x2_prime)
  #
  #       # Model Parameter
  #       rho_s = SX.sym('rho_s')
  #
  #       # State Equations
  #       dx1_ds = (rho_s - x1)/rho_s * tan(x2)
  #       dx2_ds = (rho_s - x1)/rho_s * (tan(u2)/cos(x2))
  #       f_expl = vertcat(dx1_ds, dx2_ds)
  #       f_impl = xdot - f_expl
  #
  #       self.model = types.SimpleNamespace()
  #       self.model.name = model_name
  #       self.model.x = vertcat(x1, x2)
  #       self.model.u = vertcat(u1, u2)
  #       self.model.f_impl_expr = f_impl
  #       self.model.f_expl_expr = f_expl
  #       self.model.p = rho_s
  #       self.model.xdot = xdot
  #       self.nx = self.model.x.size()[0]
  #       self.nu = self.model.u.size()[0]
  #
  #       return self.model