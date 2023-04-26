#%% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


#%% Functions

# def nominal_state_update(nominal_state, w_m, a_m, dt):
#     """
#     function to perform the nominal state update

#     :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
#                     all elements are 3x1 vectors except for q which is a Rotation object
#     :param w_m: 3x1 vector - measured angular velocity in radians per second
#     :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
#     :param dt: duration of time interval since last update in seconds
#     :return: new tuple containing the updated state
#     """
#     # Unpack nominal_state tuple
#     p, v, q, a_b, w_b, g = nominal_state

#     # YOUR CODE HERE
#     new_p = np.zeros((3, 1))
#     new_v = np.zeros((3, 1))
#     new_q = Rotation.identity()

#     return new_p, new_v, new_q, a_b, w_b, g


# def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
#                             accelerometer_noise_density, gyroscope_noise_density,
#                             accelerometer_random_walk, gyroscope_random_walk):
#     """
#     Function to update the error state covariance matrix

#     :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
#                         all elements are 3x1 vectors except for q which is a Rotation object
#     :param error_state_covariance: 18x18 initial error state covariance matrix
#     :param w_m: 3x1 vector - measured angular velocity in radians per second
#     :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
#     :param dt: duration of time interval since last update in seconds
#     :param accelerometer_noise_density: standard deviation of accelerometer noise
#     :param gyroscope_noise_density: standard deviation of gyro noise
#     :param accelerometer_random_walk: accelerometer random walk rate
#     :param gyroscope_random_walk: gyro random walk rate
#     :return:
#     """

#     # Unpack nominal_state tuple
#     p, v, q, a_b, w_b, g = nominal_state

#     # YOUR CODE HERE

#     # return an 18x18 covariance matrix
#     return np.identity(18)


# def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
#     """
#     Function to update the nominal state and the error state covariance matrix based on a single
#     observed image measurement uv, which is a projection of Pw.

#     :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
#                         all elements are 3x1 vectors except for q which is a Rotation object
#     :param error_state_covariance: 18x18 initial error state covariance matrix
#     :param uv: 2x1 vector of image measurements
#     :param Pw: 3x1 vector world coordinate
#     :param error_threshold: inlier threshold
#     :param Q: 2x2 image covariance matrix
#     :return: new_state_tuple, new error state covariance matrix
#     """
    
#     # Unpack nominal_state tuple
#     p, v, q, a_b, w_b, g = nominal_state

#     # YOUR CODE HERE - compute the innovation next state, next error_state covariance
#     innovation = np.zeros((2, 1))
#     return (p, v, q, a_b, w_b, g), error_state_covariance, innovation


def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    new_p = np.zeros((3, 1))
    new_v = np.zeros((3, 1))
    new_q = Rotation.identity()

    R     = Rotation.as_matrix(q)
    new_a = (R @ (a_m - a_b) + g)
    new_p = p + (v * dt) + (1/2) * new_a * dt ** 2
    new_v = v + new_a * dt
    new_q = q * Rotation.from_rotvec(((w_m - w_b) * dt).flatten())


    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    R = Rotation.as_matrix(q)
    Fx = np.zeros((18, 18))
    Qi = np.zeros((12, 12))
    Fi = np.zeros((18, 12))
    
    Fx[:3,:3]       = Fx[3:6, 3:6] = Fx[9:12, 9:12]  = Fx[12:15, 12:15] = Fx[15:, 15:] = np.identity(3)
    Fx[:3, 3:6]     = Fx[3:6, 15:] = np.identity(3) * dt
    Fx[6:9, 12:15]  = -np.identity(3) * dt

    a               = (a_m - a_b).flatten()
    a_skew          = np.array([[0,         -a[2],      a[1]],
                                [a[2],        0,       -a[0]],
                                [-a[1],      a[0],      0   ]])

    Fx[3:6, 6:9]    = -(R @ a_skew) * dt
    Fx[3:6, 9:12]   = -R * dt
    rot_f_vec       = Rotation.from_rotvec(((w_m - w_b)*dt).flatten())
    Fx[6:9, 6:9]    = rot_f_vec.as_matrix().T

    Fi[3:6, 0:3]    = Fi[6:9, 3:6] = Fi[9:12, 6:9] = Fi[12:15, 9:] = np.identity(3)

    Qi[0:3, 0:3]    = (accelerometer_noise_density**2) * (dt**2) * np.identity(3)
    Qi[3:6, 3:6]    = (gyroscope_noise_density**2)   * (dt**2) * np.identity(3)
    Qi[6:9, 6:9]    = (accelerometer_random_walk**2) * dt * np.identity(3)
    Qi[9:, 9:]      = (gyroscope_random_walk**2)     * dt * np.identity(3)

    # return an 18x18 covariance matrix
    return ((Fx @ error_state_covariance @ Fx.T) + (Fi @ Qi @ Fi.T))


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """
    
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE - compute the innovation next state, next error_state covariance
    # innovation = np.zeros((2,1))
    R  = Rotation.as_matrix(q)
    Pc = (R.T @ (Pw - p))
    Pc_norm = (Pc[0:2]/Pc[2]).reshape(-1,1)
    innovation = uv - Pc_norm

    if norm(innovation) < error_threshold:
        #Camera Measurements
        zt           = (Pc_norm).flatten()
        diff_zt_Pc   = (1/Pc[2]) * np.array([[1, 0, -zt[0]],
                                             [0, 1, -zt[1]]], dtype=float)
                                             
        Pc0          = (R.T @ (Pw - p)).flatten()

        diff_pc_delt = np.array([[0,       -Pc0[2],    Pc0[1]],
                                [Pc0[2],      0,      -Pc0[0]],
                                [-Pc0[1],    Pc0[0],      0  ]], dtype=float)
        
        # Jacobians
        diff_Pc_delp = - R.T
        diff_zt_delt = diff_zt_Pc @ diff_pc_delt
        diff_zt_delp = diff_zt_Pc @ diff_Pc_delp

        Ht = np.zeros((2,18))
        Ht[:, 0:3] = diff_zt_delp
        Ht[:, 6:9] = diff_zt_delt

        #Measurement Update Equations
        Kt = (error_state_covariance @ Ht.T) @ np.linalg.inv((Ht @ error_state_covariance @ Ht.T) + Q)
        error_state_covariance = (np.identity(18) - Kt @ Ht) @ error_state_covariance @ (np.identity(18) - Kt @ Ht).T + (Kt @ Q @ Kt.T)
        del_x = Kt @ innovation
        
        #Update state estimation 
        del_p  = del_x[0:3]
        p      = p + del_p

        del_v  = del_x[3:6]
        v      = v + del_v

        del_t  = del_x[6:9].reshape(-1, )
        # q*(Rotation.from_rotvec(dx[6:9]))
        q      = Rotation.from_matrix(R @ (Rotation.from_rotvec(del_t.flatten()).as_matrix()))

        del_ab = del_x[9:12]
        a_b    = a_b + del_ab

        del_wb = del_x[12:15]
        w_b    = w_b + del_wb

        del_g  = del_x[15:18]
        g      = g + del_g

        # print(p)

    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation
