import numpy as np

from scipy.sparse import  csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse import SparseWarning
from scipy.sparse import lil_matrix

import sys
sys.path.append('/home/josh/Desktop/meam620/project3/proj3/code/')

from .graph_search import graph_search

###################################################################################################
###################################### Related Works can be found in Readme  ######################

def perpendicular_distance(p, a, b):

    direction_vector = np.divide(b - a, np.linalg.norm(b - a))
    s = np.dot(a - p, direction_vector)
    t = np.dot(p - b, direction_vector)
    h = np.maximum.reduce([s, t, 0])
    c = np.cross(p - a, direction_vector)

    return np.hypot(h, np.linalg.norm(c))

def RDP(trajectory, perp_threshold = 0.05):
    start             = 0
    end               = len(trajectory)-1
    maximum_Distance  = 0 
    max_index         = 0 

    for i in range(start+1,end):
        perp_dist = perpendicular_distance(trajectory[i], trajectory[start], trajectory[end])
        if perp_dist > maximum_Distance:
            maximum_Distance = perp_dist 
            max_index        = i 

    if maximum_Distance > perp_threshold:
        l = RDP(trajectory[start:max_index+1], perp_threshold)
        r = RDP(trajectory[max_index:], perp_threshold)

        return np.vstack((l[:-1], r))

    else:
        return np.vstack((trajectory[0], trajectory[end]))

#####################################################################################################
#####################################################################################################

class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        # self.resolution = np.array([0.2, 0.2, 0.2])
        # self.margin = 0.2

        self.resolution = np.array([0.25, 0.25, 0.25])
        self.margin = 0.5

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a cla8ss member; you will
        # need it for debugging and it will be used when plotting results.
        self.points = np.zeros((1,3)) # shape=(n_pts,3)
        # print(self.path.shape)    
        self.points =  RDP(self.path)
        # self.points_new = RDP(self.path)

        # print(self.points.shape)
        # print()
        # print(self.points_new.shape)
        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE
        self.speed  = 2.8

        self.points_len = len(self.points)
        
        self.inertia          = [np.zeros(3)] 
        self.segment_velocity = [np.zeros(3)]
        self.segment_duration = [0]
        self.segment_startime = [0]
        self.distance_lst     = [0]

        for idx in range(self.points_len - 1):
            displacement = (self.points[idx+1] - self.points[idx])
            distance     = (np.linalg.norm((self.points[idx+1] - self.points[idx])))
            I            = displacement/distance
            self.inertia.append(I)
            self.distance_lst.append(distance)

            if distance < 2.5:
                self.speed = 2.8
            elif distance > 2.5 and distance < 3.0:
                self.speed = 3.65
            elif distance > 3:
                self.speed = 3.7

            self.segment_velocity.append(self.speed * I)
            self.segment_duration.append(distance/self.speed)
            self.segment_startime.append(self.segment_startime[idx] + (distance/self.speed))

        n = self.points.shape[0] -1
        B = lil_matrix((8 * n, 3))
        A = lil_matrix((8 * n, 8 * n))
        self.segment_duration = np.clip(np.array(self.segment_duration),0.25, np.inf)

        for cnt in range(n):
            B[8 * cnt + 3] = self.points[cnt]
            B[8 * cnt + 4] = self.points[cnt + 1]
            t_each = self.segment_duration[cnt+1]
            Mat = self.generate_M(t_each)

            A[[0, 1, 2], [6, 5, 4]] = [1, 2, 6]
            if cnt != (n-1):
                A[5 + 8 * cnt, 14 + 8 * cnt] = -1
                A[6 + 8 * cnt, 13 + 8 * cnt] = -2
                A[7 + 8 * cnt, 12 + 8 * cnt] = -6
                A[8 + 8 * cnt, 11 + 8 * cnt] = -24
                A[9 + 8 * cnt, 10 + 8 * cnt] = -120
                A[10 + 8 * cnt, 9 + 8 * cnt] = -720
                A[8 * cnt + 3:8 * cnt + 11, 8 * cnt:8 * cnt + 8] = Mat
            else:
                A[8 * cnt + 3:8 * cnt + 11, 8 * cnt:8 * cnt + 8] = Mat[:5, :]

            cnt += 1

        A = A.tocsc()
        self.C = spsolve(A, B).toarray()


    def generate_M(self, curr_t):
        return np.array([[0,                    0,                   0,                  0,                  0,                  0,           0,      1],
                         [curr_t ** 7,          curr_t ** 6,         curr_t ** 5,        curr_t ** 4,        curr_t ** 3,        curr_t ** 2, curr_t, 1],
                         [7    * (curr_t ** 6), 6   * (curr_t ** 5), 5 * (curr_t ** 4),  4  * (curr_t ** 3), 3 * (curr_t ** 2),  2 * curr_t,  1,      0],
                         [42   * (curr_t ** 5), 30  * (curr_t ** 4), 20 * (curr_t ** 3), 12 * (curr_t ** 2), 6 * curr_t,         2,           0,      0],
                         [210  * (curr_t ** 4), 120 * (curr_t ** 3), 60 * (curr_t ** 2), 24 * curr_t,        6,                  0,           0,      0],
                         [840  * (curr_t ** 3), 360 * (curr_t ** 2), 120 * curr_t,       24,                 0,                  0,           0,      0],
                         [2520 * (curr_t ** 2), 720 * curr_t,        120,                0,                  0,                  0,           0,      0],
                         [5040 * curr_t,        720,                 0,                  0,                  0,                  0,           0,      0]])

    

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE

        if t == 0:
            x     = self.points[0]
            x_dot = np.zeros(3)

        #If time is greater than the duration of the full trajectory, then velocity should be zero and position should be final waypoint
        elif t >= self.segment_startime[-1]:
            x     = self.points[-1]
            x_dot = np.zeros(3)
        
        else:
            for idx in range(self.points_len-1):
                if t < self.segment_startime[idx+1]:
                    t = t - self.segment_startime[idx]
                    t_arr_iter = self.generate_M(t)[1:6,:]

                    resu     = t_arr_iter @ self.C[idx * 8:(idx * 8) + 8, :]
                    x        = resu[0, :]
                    x_dot    = resu[1, :]
                    x_ddot   = resu[2, :]
                    x_dddot  = resu[3, :]
                    x_ddddot = resu[4, :]
                    break


        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
