from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
from Helper_scripts.Models import *
from casadi import *


class EKF:
    def __init__(self):
        pass

    def linearize(self):

        """ LINEARIZATION STEP """
        # Linearize the system

        # Discretize the system

    def prediction(self):

        """ PREDICTION STEP """
        # Predict the next state

        # Predict the next error covariance

    def correction(self):

        """ CORRECTION STEP """

        # Compute Kalman gain

        # Update estimate with measurement

        # Update error covariance






