""" Run CPG """

import time
import numpy as np
import matplotlib

# adapt as needed for your system
from sys import platform
if platform =="darwin":
  matplotlib.use("Qt5Agg")
else:
  matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv

ADD_CARTESIAN_PD = False
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=QuadrupedGymEnv,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP)

TEST_STEPS = int(10 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# CPG states: r, theta, r_dot, theta_dot for all 4 legs
r_hist      = np.zeros((4, TEST_STEPS))
theta_hist  = np.zeros((4, TEST_STEPS))
dr_hist     = np.zeros((4, TEST_STEPS))
dtheta_hist = np.zeros((4, TEST_STEPS))

# Robot state: 12 joint positions
joint_pos   = np.zeros((12, TEST_STEPS))

############## Sample Gains
# joint PD gains
kp=np.array([100,100,100])
kd=np.array([2,2,2])

# Cartesian PD gains
kpCartesian = np.diag([500]*3)
kdCartesian = np.diag([20]*3)

for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12) 

  # get desired foot positions from CPG 
  xs,zs = cpg.update()

  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities()

  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)

    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])

    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz)

    # Add joint PD contribution to tau for leg i (Equation 4)
    q_leg = q[3*i : 3*i+3]
    dq_leg = dq[3*i : 3*i+3]
    tau += kp * (leg_q - q_leg) + kd * (0 - dq_leg)

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get desired xyz position in leg frame (use ComputeJacobianAndPosition with the joint angles you just found above)
      _, pd = env.robot.ComputeJacobianAndPosition(i, leg_q)

      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      J, p = env.robot.ComputeJacobianAndPosition(i, q_leg)

      # Get current foot velocity in leg frame (Equation 2)
      v = J @ dq_leg

      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      vd = np.zeros(3)
      f = kpCartesian @ (pd - p) + kdCartesian @ (vd - v)
      tau += J.T @ f

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 

  # [TODO] save any CPG or robot states

##################################################### 
# PLOTS
#####################################################
# [TODO] Create your plots

# example
# fig = plt.figure()
# plt.plot(t,joint_pos[1,:], label='FR thigh')
# plt.legend()
# plt.show()