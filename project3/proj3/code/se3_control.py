import numpy as np
from scipy.spatial.transform import Rotation

# class SE3Control(object):
#     """

#     """
#     def __init__(self, quad_params):
#         """
#         This is the constructor for the SE3Control object. You may instead
#         initialize any parameters, control gain values, or private state here.

#         For grading purposes the controller is always initialized with one input
#         argument: the quadrotor's physical parameters. If you add any additional
#         input arguments for testing purposes, you must provide good default
#         values!

#         Parameters:
#             quad_params, dict with keys specified by crazyflie_params.py

#         """

#         # Quadrotor physical parameters.
#         self.mass            = quad_params['mass'] # kg
#         self.Ixx             = quad_params['Ixx']  # kg*m^2
#         self.Iyy             = quad_params['Iyy']  # kg*m^2
#         self.Izz             = quad_params['Izz']  # kg*m^2
#         self.arm_length      = quad_params['arm_length'] # meters
#         self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
#         self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
#         self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
#         self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

#         # You may define any additional constants you like including control gains.
#         self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
#         self.g = 9.81 # m/s^2

#         # STUDENT CODE HERE

#     def update(self, t, state, flat_output):
#         """
#         This function receives the current time, true state, and desired flat
#         outputs. It returns the command inputs.

#         Inputs:
#             t, present time in seconds
#             state, a dict describing the present state with keys
#                 x, position, m
#                 v, linear velocity, m/s
#                 q, quaternion [i,j,k,w]
#                 w, angular velocity, rad/s
#             flat_output, a dict describing the present desired flat outputs with keys
#                 x,        position, m
#                 x_dot,    velocity, m/s
#                 x_ddot,   acceleration, m/s**2
#                 x_dddot,  jerk, m/s**3
#                 x_ddddot, snap, m/s**4
#                 yaw,      yaw angle, rad
#                 yaw_dot,  yaw rate, rad/s

#         Outputs:
#             control_input, a dict describing the present computed control inputs with keys
#                 cmd_motor_speeds, rad/s
#                 cmd_thrust, N (for debugging and laboratory; not used by simulator)
#                 cmd_moment, N*m (for debugging; not used by simulator)
#                 cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
#         """
#         cmd_motor_speeds = np.zeros((4,))
#         cmd_thrust = 0
#         cmd_moment = np.zeros((3,))
#         cmd_q = np.zeros((4,))

#         # STUDENT CODE HERE

#         control_input = {'cmd_motor_speeds':cmd_motor_speeds,
#                          'cmd_thrust':cmd_thrust,
#                          'cmd_moment':cmd_moment,
#                          'cmd_q':cmd_q}
#         return control_input


class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE
        self.K_M = 1.5e-9
        self.K_F = 6.11e-8

        #Tunable parameters
        #Velocity = 1.7
        # self.kd  = np.diag(np.array([3, 3, 3]))
        # self.kp  = np.diag(np.array([5.5, 5.5, 5.5]))
        # self.K_R = np.diag(np.array([800, 800, 40]))
        # self.K_w = np.diag(np.array([50, 50, 10]))

        #Velocity = 2.7
        # self.kd  = np.diag(np.array([5., 5., 5.]))
        # self.kp  = np.diag(np.array([5., 5., 8.]))
        # self.K_R = np.diag(np.array([900, 900, 60]))
        # self.K_w = np.diag(np.array([100, 100, 10]))

        # self.kd  = np.diag(np.array([4.5, 4.5, 4.5]))
        # self.kp  = np.diag(np.array([6, 6, 8.]))
        # self.K_R = np.diag(np.array([900, 900, 80]))
        # self.K_w = np.diag(np.array([50, 50, 10]))

        self.kd  = np.diag(np.array([4.5, 4.5, 4.5]))
        self.kp  = np.diag(np.array([6, 6, 8.]))
        self.K_R = np.diag(np.array([900, 900, 80]))
        self.K_w = np.diag(np.array([50, 50, 10]))

        self.gamma = self.k_drag/ self.k_thrust

        self.uw_mat = np.array([[ 1,        1,          1,      1          ],
                                [ 0,  self.arm_length,  0, -self.arm_length],
                                [-self.arm_length,  0,  self.arm_length,  0],
                                [ self.gamma, -self.gamma,  self.gamma, -self.gamma]])

        self.u_inv = np.linalg.inv(self.uw_mat)


    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE
        #Eq 31
        r_ddot_des  = flat_output['x_ddot'] \
                      - (self.kd@(state['v'] - flat_output['x_dot']))\
                      - (self.kp@(state['x'] - flat_output['x']))

        #Eq 33
        F_des       = ((self.mass * r_ddot_des) + np.array([0,0,self.mass * self.g])).reshape(3,1)

        #Eq 34
        R_1         = (Rotation.from_quat(state["q"])).as_matrix()
        b3          = np.dot(R_1, np.array([[0],[0],[1]]))
        u1          = (b3.T @ F_des)[0]

        #Eq 35
        b3_des      = F_des/np.linalg.norm(F_des)

        #Eq 36
        yaw_dir     = np.array([[np.cos(flat_output['yaw'])], 
                                [np.sin(flat_output['yaw'])], 
                                [0]])

        #Eq 37
        b2_des      = np.cross(b3_des, yaw_dir, axis = 0) / np.linalg.norm(np.cross(b3_des, yaw_dir, axis = 0))

        #Eq 39
        R_des       = np.array([np.cross(b2_des,b3_des, axis = 0), b2_des, b3_des]).reshape(3,3).T

        #Eq_40
        v_op        = ((R_des.T @ R_1 ) - (R_1.T @ R_des))
        e_R         = 1/2 * np.array([v_op[2][1], v_op[0][2], v_op[1][0]])
        e_w         = state['w']
        u2          = self.inertia @ (-self.K_R @ e_R - self.K_w @ e_w)

        u           = np.array([u1[0], u2[0], u2[1], u2[2]])

        cmd_motor_speeds = np.sign((self.u_inv @ u)/self.k_thrust)\
                          *np.sqrt(np.abs((self.u_inv @ u)/self.k_thrust))   

        cmd_motor_speeds = np.clip(cmd_motor_speeds,self.rotor_speed_min,self.rotor_speed_max)
        cmd_thrust       = u1
        cmd_moment       = u2

        cmd_q            = Rotation.from_matrix(R_des).as_quat()

        
        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}

        return control_input
