""" Run CPG """

import time
import numpy as np
import matplotlib
from sys import platform
import json
def load_params(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
params = load_params("params_bound.json")
GAIT_NAME = params["cpg_params"]["gait"]

# adapt as needed for your system
if platform == "darwin":
    matplotlib.use("Qt5Agg")
else:
    matplotlib.use("TkAgg")

from matplotlib import pyplot as plt
from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv

ADD_CARTESIAN_PD = True
TIME_STEP = 0.001

foot_y = 0.0838  # this is the hip length
sideSign = np.array([-1, 1, -1, 1])  # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(
    render=True,            # visualize
    on_rack=False,          # useful for debugging!
    isRLGymInterface=False, # not using RL
    time_step=TIME_STEP,
    action_repeat=1,
    motor_control_mode="TORQUE",
    add_noise=False,        # ideal conditions
)

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP)
# cpg._set_gait("TROT")

TEST_STEPS = int(10 / TIME_STEP)
t = np.arange(TEST_STEPS) * TIME_STEP

# CPG histories
r_hist      = np.zeros((4, TEST_STEPS))
theta_hist  = np.zeros((4, TEST_STEPS))
dr_hist     = np.zeros((4, TEST_STEPS))
dtheta_hist = np.zeros((4, TEST_STEPS))

# Robot state histories
joint_pos = np.zeros((12, TEST_STEPS))

# joint PD gains
kp_joint = np.array(params["joint_pd"]["kp_joint"])
kd_joint = np.array(params["joint_pd"]["kd_joint"])
# Cartesian PD gains
kp_front = np.array(params["cartesian_pd"]["kp_front"])
kd_front = np.array(params["cartesian_pd"]["kd_front"])
kp_hind  = np.array(params["cartesian_pd"]["kp_hind"])
kd_hind  = np.array(params["cartesian_pd"]["kd_hind"])


# Desired vs actual foot trajectory (leg 0)
des_x = np.zeros(TEST_STEPS)
des_y = np.zeros(TEST_STEPS)
des_z = np.zeros(TEST_STEPS)

act_x = np.zeros(TEST_STEPS)
act_y = np.zeros(TEST_STEPS)
act_z = np.zeros(TEST_STEPS)

# Joint desired/actual histories
q_des_hist = np.zeros((12, TEST_STEPS))
q_act_hist = np.zeros((12, TEST_STEPS))

# Base motion
base_pos = np.zeros((TEST_STEPS, 3))
base_vel = np.zeros((TEST_STEPS, 3))
forward_hist = np.zeros((TEST_STEPS, 1))
contact_hist = np.zeros((TEST_STEPS, 4))
power_hist = np.zeros((TEST_STEPS, 1))

# ------------------------ SIM LOOP ------------------------
for j in range(TEST_STEPS):

    action = np.zeros(12)

    # CPG desired positions
    xs, zs = cpg.update()

    des_x[j] = xs[0]
    des_z[j] = zs[0]

    # Actual foot position
    J, p = env.robot.ComputeJacobianAndPosition(0)
    act_x[j] = p[0]
    act_y[j] = p[1]
    act_z[j] = p[2]

    # Record CPG states
    r_hist[:, j]      = cpg.get_r()
    theta_hist[:, j]  = cpg.get_theta()
    dr_hist[:, j]     = cpg.get_dr()
    dtheta_hist[:, j] = cpg.get_dtheta()

    # Joint states
    q  = env.robot.GetMotorAngles()
    dq = env.robot.GetMotorVelocities()

    # Base states
    pos = env.robot.GetBasePosition()
    vel = env.robot.GetBaseLinearVelocity()
    base_pos[j, :] = pos
    base_vel[j, :] = vel

    R = env.robot.GetBaseOrientationMatrix()
    v_body = R.T @ vel
    forward_hist[j] = v_body[0]

    # Horizontal body-frame speed (or use world frame if you prefer) 
    v_xy_body = np.sqrt(v_body[0]**2 + v_body[1]**2) 
    # Or world-frame horizontal speed: 
    v_xy_world = np.sqrt(vel[0]**2 + vel[1]**2) 
    # Store one of them (pick one and use consistently) 
    if j == 0: 
      speed_xy_hist = np.zeros(TEST_STEPS) 
    speed_xy_hist[j] = v_xy_world # or v_xy_body

    _, _, _, contacts = env.robot.GetContactInfo()
    contact_hist[j, :] = contacts

    # --------------- PER-LEG CONTROL ---------------
    for i in range(4):

        tau = np.zeros(3)

        leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])

        leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz)

        q_leg = q[3*i : 3*i+3]
        dq_leg = dq[3*i : 3*i+3]

        # Soft-start: reduce knee gain for first 250 steps
        if j <= 250:
            kp = kp_joint.copy()
            kp[2] = kp[2] / 4
            kd = kd_joint
        else:
            kp = kp_joint.copy()
            kd = kd_joint

        tau += kp * (leg_q - q_leg) + kd * (-dq_leg)

        q_des_hist[3*i:3*i+3, j] = leg_q
        q_act_hist[3*i:3*i+3, j] = q_leg

        des_y[j] = sideSign[0] * foot_y

        # -------- log desired and actual joint angles ---------- 
        q_des_hist[3*i:3*i+3, j] = leg_q 
        q_act_hist[3*i:3*i+3, j] = q_leg 
        # -------- RECORD DESIRED POSITIONS (leg 0) ---------- 
        des_x[j] = xs[0] 
        des_z[j] = zs[0] 
        des_y[j] = sideSign[0] * foot_y 
        # -------- RECORD ACTUAL FOOT POSITION ---------- 
        J, p = env.robot.ComputeJacobianAndPosition(0) # leg 0 
        act_x[j] = p[0] 
        act_z[j] = p[2]

        # Cartesian PD
        if ADD_CARTESIAN_PD:
            J, p = env.robot.ComputeJacobianAndPosition(i)
            v = J @ dq_leg

            des_p = leg_xyz
            des_v = np.zeros(3)

            if i in [0, 1]:
                kpCartesian = kp_front
                kdCartesian = kd_front
            else:
                kpCartesian = kp_hind
                kdCartesian = kd_hind

            tau += J.T @ (kpCartesian @ (des_p - p) + kdCartesian @ (des_v - v))

        action[3*i : 3*i+3] = tau

    # Step simulation
    env.step(action)

    tau = action
    qdot = env.robot.GetMotorVelocities()
    power_hist[j] = np.sum(np.abs(tau * qdot))


# ============================================================
#                        PLOTS
# ============================================================

discard = 150

# ----------------- CPG STATES -----------------
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
leg_names = ["FR", "FL", "RR", "RL"]

for i in range(4):
    axs[i].plot(t[discard:], r_hist[i, discard:], label="r")
    axs[i].plot(t[discard:], theta_hist[i, discard:], label="theta")
    axs[i].plot(t[discard:], dr_hist[i, discard:], label="r_dot")
    axs[i].plot(t[discard:], dtheta_hist[i, discard:], label="theta_dot")
    axs[i].set_ylabel(f"Leg {leg_names[i]}")
    axs[i].legend(loc="upper right")
    axs[i].grid(True)

axs[-1].set_xlabel("Time (s)")
plt.suptitle(f"CPG States for {GAIT_NAME} Gait (r, θ, ṙ, θ̇) — After Transient Removed")
plt.tight_layout()
plt.show()

# ----------------- FOOT TRAJECTORY -----------------
plt.figure(figsize=(10, 6))
plt.plot(t, des_x, 'r--', label="Desired x(t)")
plt.plot(t, act_x, 'r',   label="Actual x(t)")
plt.plot(t, des_y, 'g--', label="Desired y(t)")
plt.plot(t, act_y, 'g',   label="Actual y(t)")
plt.plot(t, des_z, 'b--', label="Desired z(t)")
plt.plot(t, act_z, 'b',   label="Actual z(t)")
plt.xlabel("Time (s)")
plt.ylabel("Foot Position (m)")
plt.title(f"Desired vs Actual Foot Trajectory (Leg 0, {GAIT_NAME} Gait)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------- JOINT ANGLES (LEG 0) -----------------
leg_idx = 0
start = 3 * leg_idx

fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
joint_names = ["Joint 0", "Joint 1", "Joint 2"]

for k in range(3):
    ax[k].plot(t, q_des_hist[start+k], 'r--', label="Desired")
    ax[k].plot(t, q_act_hist[start+k], 'b',   label="Actual")
    ax[k].set_ylabel(joint_names[k])
    ax[k].grid(True)
    ax[k].legend(loc="upper right")

ax[-1].set_xlabel("Time (s)")
fig.suptitle(f"Desired vs Actual Joint Angles (Leg 0, {GAIT_NAME} Gait)")
plt.tight_layout()
plt.show()

# ----------------- PERFORMANCE METRICS -----------------
discard = int(0.5 * TEST_STEPS)

mean_vx  = np.mean(forward_hist[discard:])
mean_vxy = np.mean(speed_xy_hist[discard:])

print("Mean forward body velocity v_x:", mean_vx, "m/s")
print("Mean horizontal speed v_xy:",     mean_vxy, "m/s")

duty_cycles = np.mean(contact_hist[discard:], axis=0)
print("Duty cycle per leg:", duty_cycles)
print("Average duty cycle:", np.mean(duty_cycles))

# ----------------- STEP TIMES -----------------
stance_times = [[] for _ in range(4)]
swing_times  = [[] for _ in range(4)]
last_contact_change = [-1] * 4

for leg in range(4):
    for j in range(discard + 1, TEST_STEPS):
        if contact_hist[j-1, leg] == 1 and contact_hist[j, leg] == 0:
            stance_times[leg].append((j - last_contact_change[leg]) * TIME_STEP)
            last_contact_change[leg] = j

        if contact_hist[j-1, leg] == 0 and contact_hist[j, leg] == 1:
            swing_times[leg].append((j - last_contact_change[leg]) * TIME_STEP)
            last_contact_change[leg] = j

print("Mean stance times:", [np.mean(s) for s in stance_times])
print("Mean swing times:",  [np.mean(s) for s in swing_times])

# ----------------- COST OF TRANSPORT -----------------
distance_xy_int = np.sum(speed_xy_hist[discard:] * TIME_STEP)
mass = sum(env.robot.GetTotalMassFromURDF())

CoT_xy_int = np.sum(power_hist[discard:] * TIME_STEP) / (mass * 9.81 * distance_xy_int)

print("Integrated horizontal distance:", distance_xy_int, "m")
print("CoT (using integrated path length):", CoT_xy_int)
