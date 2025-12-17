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

ADD_CARTESIAN_PD = True
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP)
# cpg._set_gait("TROT")

TEST_STEPS = int(10 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# CPG states: r, theta, r_dot, theta_dot for all 4 legs
r_hist      = np.zeros((4, TEST_STEPS))
theta_hist  = np.zeros((4, TEST_STEPS))
dr_hist     = np.zeros((4, TEST_STEPS))
dtheta_hist = np.zeros((4, TEST_STEPS))

# Robot state: 12 joint positions
joint_pos   = np.zeros((12, TEST_STEPS))

base_pos_hist = np.zeros((3, TEST_STEPS))
base_vel_hist = np.zeros((3, TEST_STEPS))
motor_tau_hist = np.zeros((12, TEST_STEPS))
motor_dq_hist = np.zeros((12, TEST_STEPS))

############## Sample Gains
# joint PD gains
kp=np.array([100,100,100])
kd=np.array([2,2,2])

# Cartesian PD gains
kpCartesian = np.diag([500]*3)
kdCartesian = np.diag([20]*3)

des_foot_xyz = np.zeros((3, TEST_STEPS))
act_foot_xyz = np.zeros((3, TEST_STEPS))
des_leg_q = np.zeros((3, TEST_STEPS))
act_leg_q = np.zeros((3, TEST_STEPS))

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
      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      J, p = env.robot.ComputeJacobianAndPosition(i)

      # Get current foot velocity in leg frame (Equation 2)
      v = J @ dq_leg

      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      des_p  = leg_xyz
      des_v = np.zeros(3)
      tau += J.T @ ((kpCartesian @ (des_p - p) + kdCartesian @ (des_v - v))) 

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 

  # [TODO] save any CPG or robot states
  r_hist[:, j] = cpg.get_r()
  theta_hist[:, j] = cpg.get_theta()
  dr_hist[:, j] = cpg.get_dr()
  dtheta_hist[:, j] = cpg.get_dtheta()
  joint_pos[:, j] = q

  i0 = 0
  leg_xyz0 = np.array([xs[i0], sideSign[i0] * foot_y, zs[i0]])
  leg_q0 = env.robot.ComputeInverseKinematics(i0, leg_xyz0)
  J0, p0 = env.robot.ComputeJacobianAndPosition(i0)

  des_foot_xyz[:, j] = leg_xyz0
  act_foot_xyz[:, j] = p0
  des_leg_q[:, j] = leg_q0
  act_leg_q[:, j] = q[0:3]

  base_pos_hist[:, j] = np.array(env.robot.GetBasePosition())
  base_vel_hist[:, j] = np.array(env.robot.GetBaseLinearVelocity())
  motor_tau_hist[:, j] = np.array(env.robot.GetMotorTorques())
  motor_dq_hist[:, j] = np.array(env.robot.GetMotorVelocities())

##################################################### 
# PLOTS
#####################################################
transient_s = 0.5
s = int(transient_s / TIME_STEP)
e = TEST_STEPS
tt = t[s:e]

leg_names = ["Front Right (FR)", "Front Left (FL)", "Rear Right (RR)", "Rear Left (RL)"]

fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
for i in range(4):
  axs[i].plot(tt, r_hist[i, s:e], label=r"$r(t)$")
  axs[i].plot(tt, theta_hist[i, s:e], label=r"$\theta(t)$")
  axs[i].plot(tt, dr_hist[i, s:e], label=r"$\dot r(t)$")
  axs[i].plot(tt, dtheta_hist[i, s:e], label=r"$\dot\theta(t)$")
  axs[i].set_ylabel(f"{leg_names[i]}")
  axs[i].legend()
  axs[i].grid(which='both')
axs[-1].set_xlabel(r"$t\,[\mathrm{s}]$")
fig.tight_layout()

fig2 = plt.figure(figsize=(14, 6))
plt.plot(t, des_foot_xyz[0, :], "--", label=r"$x^\mathrm{ref}(t)$")
plt.plot(t, act_foot_xyz[0, :], label=r"$x(t)$")
plt.plot(t, des_foot_xyz[1, :], "--", label=r"$y^\mathrm{ref}(t)$")
plt.plot(t, act_foot_xyz[1, :], label=r"$y(t)$")
plt.plot(t, des_foot_xyz[2, :], "--", label=r"$z^\mathrm{ref}(t)$")
plt.plot(t, act_foot_xyz[2, :], label=r"$z(t)$")
plt.xlabel(r"$t\,[\mathrm{s}]$")
plt.ylabel(r"$p(t)\,[\mathrm{m}]$")
plt.legend()
plt.grid(which='both')
plt.tight_layout()

fig3, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
axs[0].plot(t, des_leg_q[0, :], "--", label=r"$q_{0}^\mathrm{ref}(t)$")
axs[0].plot(t, act_leg_q[0, :], label=r"$q_0(t)$")
axs[0].set_ylabel(r"$q_0(t)\,[\mathrm{rad}]$")
axs[0].legend()
axs[0].grid(which='both')
axs[1].plot(t, des_leg_q[1, :], "--", label=r"$q_{1}^\mathrm{ref}(t)$")
axs[1].plot(t, act_leg_q[1, :], label=r"$q_1(t)$")
axs[1].set_ylabel(r"$q_1(t)\,[\mathrm{rad}]$")
axs[1].legend()
axs[1].grid(which='both')
axs[2].plot(t, des_leg_q[2, :], "--", label=r"$q_{2}^\mathrm{ref}(t)$")
axs[2].plot(t, act_leg_q[2, :], label=r"$q_2(t)$")
axs[2].set_ylabel(r"$q_2(t)\,[\mathrm{rad}]$")
axs[2].set_xlabel(r"$t\,[\mathrm{s}]$")
axs[2].legend()
axs[2].grid(which='both')
fig3.align_ylabels(axs)
fig3.tight_layout()

vx = base_vel_hist[0, s:e]
vmax = float(np.max(vx))
vmin = float(np.min(vx))

theta0 = theta_hist[0, s:e]
wrap_idxs = np.where(theta0[1:] < theta0[:-1])[0] + 1
stride_times = []
stance_times = []
swing_times = []
if wrap_idxs.size >= 2:
  for k in range(wrap_idxs.size - 1):
    a = wrap_idxs[k]
    b = wrap_idxs[k+1]
    stride = (b - a) * TIME_STEP
    if stride <= 0:
      continue
    stance = float(np.sum(theta0[a:b] > np.pi)) * TIME_STEP
    swing = stride - stance
    stride_times.append(stride)
    stance_times.append(stance)
    swing_times.append(swing)

if len(stride_times) > 0:
  Tstride = float(np.mean(stride_times))
  Tstance = float(np.mean(stance_times))
  Tswing = float(np.mean(swing_times))
  duty_ratio = float(Tstance / Tstride) if Tstride > 0 else float("nan")
else:
  Tstride = float("nan")
  Tstance = float("nan")
  Tswing = float("nan")
  duty_ratio = float("nan")

power = np.sum(np.abs(motor_tau_hist[:, s:e] * motor_dq_hist[:, s:e]), axis=0)
energy = float(np.sum(power) * TIME_STEP)
dx = float(base_pos_hist[0, e-1] - base_pos_hist[0, s])
mass = float(np.sum(env.robot.GetTotalMassFromURDF()))
cot = float(energy / (mass * 9.81 * dx)) if dx > 1e-8 else float("inf")

print("====================================================")
print(f"Body velocity x: min = {vmin:.6f} m/s, max = {vmax:.6f} m/s")
print(f"Duty ratio D = {duty_ratio:.6f}")
print(f"Stride time T_stride = {Tstride:.6f} s")
print(f"Stance time T_stance = {Tstance:.6f} s")
print(f"Swing time  T_swing  = {Tswing:.6f} s")
print(f"Cost of Transport (CoT) = {cot:.6f}")
print("====================================================")

plt.show()
env.close()
