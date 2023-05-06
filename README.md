# Autonomous VIO based Quadcopter
<p>Implemented a planning and control stack to track trajectories in a simulation environment along with state estimation to estimate the state of the robot through an obstacle filled environment.</p>
<p>Performance was measured in terms of in-simulation flight time from start to goal. </p>

## Usage:
<p>The util directory contains a few maps that you can use for testing your planner, controller and
estimator. </p>



## Performance:

### Figure 3: Trajectory optimization for 3 different trajectories on a single map
<img src=project3/Images/1.png > <p></p>
<img src=project3/Images/2.png > <p></p>
<img src=project3/Images/3.png > <p></p>

#### Figure 4: Dense Waypoints (without any optimiziation)
<img src=project3/Images/4.png > <p></p>

#### Figure 5: Optimized trajectory using RDP algorithm along with Minimum Snap
<img src=project3/Images/5.png > <p></p>

