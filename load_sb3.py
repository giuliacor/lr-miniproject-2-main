import os, sys
import gymnasium as gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
if platform =="darwin": # mac
  import PyQt5
  matplotlib.use("Qt5Agg")
else: # linux
  matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

# utils
from env.quadruped_gym_env import QuadrupedGymEnv
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results

LEARNING_ALG = "PPO" #"SAC"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '102824115106'
log_dir = interm_dir + 'slopes/SLOPES_seed0_CPG_1M/'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
# initialize env configs (render at test time)
env_config = {}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False

env_config["motor_control_mode"] = "CPG"
env_config["task_env"] = "LR_COURSE_TASK"
env_config["observation_space_mode"] = "LR_COURSE_OBS"
env_config["on_rack"] = False
env_config["terrain"] = "SLOPES"

is_vel_task = ("/cpg/VEL_" in log_dir.replace("\\", "/")) or ("\\cpg\\VEL_" in log_dir)

def moving_average_valid(x, window):
    x = np.asarray(x, dtype=float)
    if window <= 1 or x.size == 0:
        return x
    w = int(window)
    if x.size < w:
        return np.full_like(x, np.mean(x))
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")

def moving_average_same(x, window):
    x = np.asarray(x, dtype=float)
    if window <= 1 or x.size == 0:
        return x
    w = int(window)
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")

def load_curve_one_seed(seed_dir, ma_window_episodes=50):
    df = load_results(seed_dir)
    if df is None or len(df) == 0:
        return None
    r = np.asarray(df["r"], dtype=float)
    l = np.asarray(df["l"], dtype=float)
    t = np.cumsum(l)
    r_ma = moving_average_valid(r, ma_window_episodes)
    l_ma = moving_average_valid(l, ma_window_episodes)
    t_ma = t[(ma_window_episodes - 1):] if (ma_window_episodes > 1 and t.size >= ma_window_episodes) else t
    n = min(t_ma.size, r_ma.size, l_ma.size)
    return t_ma[:n], r_ma[:n], l_ma[:n]

def make_seed_dir(base_dir, s):
    out = base_dir
    for k in range(5):
        out = out.replace(f"seed{k}", f"seed{s}")
    return out

seed_dirs = []
for s in range(5):
    d = make_seed_dir(log_dir, s)
    if os.path.isdir(d):
        seed_dirs.append((s, d))

curves = []
for (s, d) in seed_dirs:
    try:
        c = load_curve_one_seed(d, ma_window_episodes=50)
        if c is not None:
            curves.append((s, d, c[0], c[1], c[2]))
    except Exception as e:
        print("Skipping", d, "due to error:", e)

if len(curves) > 0:
    t_max_common = min([float(np.max(t)) for (_, _, t, _, _) in curves if t.size > 0])
    grid = np.linspace(0.0, t_max_common, 800)

    rew_stack = []
    len_stack = []

    fig, axs = plt.subplots(2, 1, figsize=(10, 7.0), sharex=True)

    seed_lines_0 = []
    for (s, d, t, r, l) in curves:
        r_i = np.interp(grid, t, r)
        rew_stack.append(r_i)
        line, = axs[0].plot(grid, r_i, alpha=0.35, label=f"seed {s}")
        seed_lines_0.append((s, line))
    rew_stack = np.asarray(rew_stack)
    rew_mean = np.mean(rew_stack, axis=0)
    rew_std = np.std(rew_stack, axis=0)
    mean_line_0, = axs[0].plot(grid, rew_mean, linewidth=2.5, label="mean")
    std_poly_0 = axs[0].fill_between(grid, rew_mean - rew_std, rew_mean + rew_std, alpha=0.2, label=r"$\pm 1\sigma$")
    axs[0].set_ylabel(r"Mean Episode Reward")
    # axs[0].set_title(LEARNING_ALG + r" Training Curves")
    axs[0].grid(True, alpha=0.3)

    seed_lines_1 = []
    for (s, d, t, r, l) in curves:
        l_i = np.interp(grid, t, l)
        len_stack.append(l_i)
        line, = axs[1].plot(grid, l_i, alpha=0.35, label=f"seed {s}")
        seed_lines_1.append((s, line))
    len_stack = np.asarray(len_stack)
    len_mean = np.mean(len_stack, axis=0)
    len_std = np.std(len_stack, axis=0)
    mean_line_1, = axs[1].plot(grid, len_mean, linewidth=2.5, label="mean")
    std_poly_1 = axs[1].fill_between(grid, len_mean - len_std, len_mean + len_std, alpha=0.2, label=r"$\pm 1\sigma$")
    axs[1].set_xlabel(r"Timesteps")
    axs[1].set_ylabel(r"Mean Episode Length")
    axs[1].grid(True, alpha=0.3)

    seed_lines_0_sorted = [line for (_, line) in sorted(seed_lines_0, key=lambda x: x[0])]
    seed_labels_0_sorted = [f"seed {s}" for (s, _) in sorted(seed_lines_0, key=lambda x: x[0])]
    handles0 = seed_lines_0_sorted + [mean_line_0]
    labels0 = seed_labels_0_sorted + ["mean"]
    leg0 = axs[0].legend(handles0, labels0, ncol=3, fontsize=9, frameon=True, loc="upper left")
    axs[0].add_artist(leg0)
    axs[0].legend([std_poly_0], [r"$\pm 1\sigma$"], fontsize=9, frameon=True, loc="upper left", bbox_to_anchor=(0.0, 0.72))

    seed_lines_1_sorted = [line for (_, line) in sorted(seed_lines_1, key=lambda x: x[0])]
    seed_labels_1_sorted = [f"seed {s}" for (s, _) in sorted(seed_lines_1, key=lambda x: x[0])]
    handles1 = seed_lines_1_sorted + [mean_line_1]
    labels1 = seed_labels_1_sorted + ["mean"]
    leg1 = axs[1].legend(handles1, labels1, ncol=3, fontsize=9, frameon=True, loc="lower right")
    axs[1].add_artist(leg1)
    axs[1].legend([std_poly_1], [r"$\pm 1\sigma$"], fontsize=9, frameon=True, loc="lower right", bbox_to_anchor=(1.0, 0.33))

    fig.align_ylabels(axs)
    plt.tight_layout()
    plt.show()
else:
    print("No valid seed directories found for plotting.")

def load_env_and_model_for_seed(seed_dir):
    stats_path = os.path.join(seed_dir, "vec_normalize.pkl")
    model_name = get_latest_model(seed_dir)

    env = lambda: QuadrupedGymEnv(**env_config)
    env = make_vec_env(env, n_envs=1)
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    if LEARNING_ALG == "PPO":
        model = PPO.load(model_name, env)
    elif LEARNING_ALG == "SAC":
        model = SAC.load(model_name, env)
    else:
        raise ValueError("Unknown LEARNING_ALG")

    return env, model, model_name

def rollout_one_seed(seed_dir, n_steps=2000, deterministic=True):
    env, model, model_name = load_env_and_model_for_seed(seed_dir)

    obs = env.reset()
    t_hist = []
    vx_hist = []
    roll_hist = []
    pitch_hist = []

    base_pos_hist = []
    motor_tau_hist = []
    motor_dq_hist = []
    contact_hist = []

    # use true env-step duration: dt_env = time_step * action_repeat
    dt_sim = None
    if hasattr(env.envs[0].env, "time_step"):
        dt_sim = float(env.envs[0].env.time_step)
    elif hasattr(env.envs[0].env, "_time_step"):
        dt_sim = float(env.envs[0].env._time_step)
    else:
        dt_sim = 0.01

    if hasattr(env.envs[0].env, "_action_repeat"):
        ar = int(env.envs[0].env._action_repeat)
    else:
        ar = 1

    dt = float(dt_sim * ar)

    # run exactly one full episode (until done), but cap with n_steps as a safety fallback
    if hasattr(env.envs[0].env, "_MAX_EP_LEN"):
        max_ep_len = float(env.envs[0].env._MAX_EP_LEN)
        n_steps = int(np.ceil(max_ep_len / dt)) + 5

    for i in range(n_steps):
        action, _states = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, info = env.step(action)

        robot = env.envs[0].env.robot
        lin_vel = robot.GetBaseLinearVelocity()
        rpy = robot.GetBaseOrientationRollPitchYaw()
        base_pos = robot.GetBasePosition()

        t_hist.append(i * dt)
        vx_hist.append(float(lin_vel[0]))
        roll_hist.append(float(rpy[0]))
        pitch_hist.append(float(rpy[1]))

        base_pos_hist.append(np.array(base_pos))
        motor_tau_hist.append(np.array(robot.GetMotorTorques()))
        motor_dq_hist.append(np.array(robot.GetMotorVelocities()))

        _, _, _, feetInContactBool = robot.GetContactInfo()
        contact_hist.append(np.array(feetInContactBool, dtype=int))

        # stop after the first completed episode (termination OR truncation)
        if dones:
            break

    try:
        mass = float(np.sum(env.envs[0].env.robot.GetTotalMassFromURDF()))
    except Exception:
        mass = None

    env.close()

    out = {
        "model_name": model_name,
        "dt": dt,
        "t": np.asarray(t_hist, dtype=float),
        "vx": np.asarray(vx_hist, dtype=float),
        "roll": np.asarray(roll_hist, dtype=float),
        "pitch": np.asarray(pitch_hist, dtype=float),
        "base_pos": np.asarray(base_pos_hist, dtype=float).T,
        "tau": np.asarray(motor_tau_hist, dtype=float).T,
        "dq": np.asarray(motor_dq_hist, dtype=float).T,
        "contact": np.asarray(contact_hist, dtype=int).T,
    }
    if mass is not None:
        out["mass"] = np.array([mass], dtype=float)
    return out

def compute_metrics(tr, transient_s=0.5):
    t = tr["t"]
    dt = float(tr["dt"])
    s = int(transient_s / dt)
    e = int(t.size)

    vx = tr["vx"][s:e]
    vmin = float(np.min(vx)) if vx.size > 0 else float("nan")
    vmax = float(np.max(vx)) if vx.size > 0 else float("nan")

    power = np.sum(np.abs(tr["tau"][:, s:e] * tr["dq"][:, s:e]), axis=0)
    energy = float(np.sum(power) * dt)

    dx = float(tr["base_pos"][0, e-1] - tr["base_pos"][0, s]) if e - 1 >= 0 else 0.0
    mass = float(np.sum(tr.get("mass", np.array([0.0])))) if "mass" in tr else None

    if mass is None:
        cot = float("nan")
    else:
        cot = float(energy / (mass * 9.81 * dx)) if dx > 1e-8 else float("inf")

    c0 = tr["contact"][0, s:e].astype(int)
    td_idxs = np.where((c0[1:] == 1) & (c0[:-1] == 0))[0] + 1

    stride_times = []
    stance_times = []
    swing_times = []
    if td_idxs.size >= 2:
        for k in range(td_idxs.size - 1):
            a = td_idxs[k]
            b = td_idxs[k+1]
            stride = (b - a) * dt
            if stride <= 0:
                continue
            stance = float(np.sum(c0[a:b] == 1)) * dt
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

    return {
        "vmin": vmin,
        "vmax": vmax,
        "Tstride": Tstride,
        "Tstance": Tstance,
        "Tswing": Tswing,
        "duty": duty_ratio,
        "cot": cot,
    }

seed_rollouts = []
for (s, d) in seed_dirs:
    try:
        tr = rollout_one_seed(d, n_steps=2000, deterministic=True)
        seed_rollouts.append((s, d, tr))
    except Exception as e:
        print("Skipping rollout for", d, "due to error:", e)

if len(seed_rollouts) == 0:
    print("No rollouts available for evaluation plots.")
else:
    dt_common = float(np.median([r["dt"] for (_, _, r) in seed_rollouts]))

    try:
        tmp_env = lambda: QuadrupedGymEnv(**env_config)
        tmp_env = make_vec_env(tmp_env, n_envs=1)
        T_plot = float(tmp_env.envs[0].env._MAX_EP_LEN) if hasattr(tmp_env.envs[0].env, "_MAX_EP_LEN") else float(np.max([r["t"][-1] for (_, _, r) in seed_rollouts]))
        tmp_env.close()
    except Exception:
        T_plot = float(np.max([r["t"][-1] for (_, _, r) in seed_rollouts]))

    t_grid = np.arange(0.0, T_plot + 1e-12, dt_common)

    def interp_to_grid_nan(tr, key):
        t = tr["t"]
        y = tr[key]
        out = np.full_like(t_grid, np.nan, dtype=float)
        if t.size == 0:
            return out
        t_end = float(t[-1])
        m = t_grid <= t_end
        out[m] = np.interp(t_grid[m], t, y)
        return out

    vx_stack = []
    roll_stack = []
    pitch_stack = []

    for (s, d, tr) in seed_rollouts:
        vx_stack.append(interp_to_grid_nan(tr, "vx"))
        roll_stack.append(interp_to_grid_nan(tr, "roll"))
        pitch_stack.append(interp_to_grid_nan(tr, "pitch"))

    vx_stack = np.asarray(vx_stack, dtype=float)
    roll_stack = np.asarray(roll_stack, dtype=float)
    pitch_stack = np.asarray(pitch_stack, dtype=float)

    vx_mean = np.nanmean(vx_stack, axis=0)
    vx_std = np.nanstd(vx_stack, axis=0)
    roll_mean = np.nanmean(roll_stack, axis=0)
    roll_std = np.nanstd(roll_stack, axis=0)
    pitch_mean = np.nanmean(pitch_stack, axis=0)
    pitch_std = np.nanstd(pitch_stack, axis=0)

    smooth_window_sec = 0.25
    smooth_window = max(1, int(round(smooth_window_sec / dt_common)))

    vx_mean_s = moving_average_same(vx_mean, smooth_window)
    roll_mean_s = moving_average_same(roll_mean, smooth_window)
    pitch_mean_s = moving_average_same(pitch_mean, smooth_window)

    vx_avg = float(np.nanmean(vx_mean))
    roll_avg = float(np.nanmean(roll_mean))
    pitch_avg = float(np.nanmean(pitch_mean))

    metrics = []
    for (s, d, tr) in seed_rollouts:
        m = compute_metrics(tr, transient_s=0.5)
        metrics.append((s, m))

    def stat(vals):
        a = np.asarray(vals, dtype=float)
        return float(np.nanmean(a)), float(np.nanstd(a))

    vmin_mean, vmin_std = stat([m["vmin"] for (_, m) in metrics])
    vmax_mean, vmax_std = stat([m["vmax"] for (_, m) in metrics])
    Tstride_mean, Tstride_std = stat([m["Tstride"] for (_, m) in metrics])
    Tstance_mean, Tstance_std = stat([m["Tstance"] for (_, m) in metrics])
    Tswing_mean, Tswing_std = stat([m["Tswing"] for (_, m) in metrics])
    duty_mean, duty_std = stat([m["duty"] for (_, m) in metrics])
    cot_mean, cot_std = stat([m["cot"] for (_, m) in metrics])

    for (s, m) in metrics:
        print("====================================================")
        print(f"seed {s}")
        print(f"Body velocity x: min = {m['vmin']:.6f} m/s, max = {m['vmax']:.6f} m/s")
        print(f"Duty ratio D = {m['duty']:.6f}")
        print(f"Stride time T_stride = {m['Tstride']:.6f} s")
        print(f"Stance time T_stance = {m['Tstance']:.6f} s")
        print(f"Swing time  T_swing  = {m['Tswing']:.6f} s")
        print(f"Cost of Transport (CoT) = {m['cot']:.6f}")
        print("====================================================")

    print("====================================================")
    print("Across seeds (mean ± std)")
    print(f"vmin = {vmin_mean:.6f} ± {vmin_std:.6f} m/s")
    print(f"vmax = {vmax_mean:.6f} ± {vmax_std:.6f} m/s")
    print(f"Duty ratio D = {duty_mean:.6f} ± {duty_std:.6f}")
    print(f"T_stride = {Tstride_mean:.6f} ± {Tstride_std:.6f} s")
    print(f"T_stance = {Tstance_mean:.6f} ± {Tstance_std:.6f} s")
    print(f"T_swing  = {Tswing_mean:.6f} ± {Tswing_std:.6f} s")
    print(f"CoT = {cot_mean:.6f} ± {cot_std:.6f}")
    print("====================================================")

    if is_vel_task:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3.6), sharex=True)

        seed_lines = []
        for i, (s, d, tr) in enumerate(seed_rollouts):
            line, = ax.plot(t_grid, vx_stack[i], alpha=0.25, label=f"seed {s}")
            seed_lines.append((s, line))

        mean_line, = ax.plot(t_grid, vx_mean, linewidth=2.5, label="mean")
        smooth_line, = ax.plot(t_grid, vx_mean_s, linewidth=2.5, label="smoothed mean")
        std_poly = ax.fill_between(t_grid, vx_mean - vx_std, vx_mean + vx_std, alpha=0.2, label=r"$\pm 1\sigma$")
        avg_line = ax.axhline(vx_avg, linestyle="--", linewidth=1.5, label=rf"avg = {vx_avg:.2f}")

        ax.set_ylabel(r"$v_x\,[\mathrm{m/s}]$")
        ax.set_xlabel(r"$t\,[\mathrm{s}]$")
        # ax.set_title(r"Base Forward Velocity")
        ax.grid(True, alpha=0.3)

        seed_lines_sorted = [line for (_, line) in sorted(seed_lines, key=lambda x: x[0])]
        seed_labels_sorted = [f"seed {s}" for (s, _) in sorted(seed_lines, key=lambda x: x[0])]
        handles = seed_lines_sorted + [mean_line, smooth_line, avg_line]
        labels = seed_labels_sorted + ["mean", "smoothed mean", f"avg = {vx_avg:.2f}"]
        leg_main = ax.legend(handles, labels, ncol=3, fontsize=9, frameon=True, loc="upper left")
        ax.add_artist(leg_main)
        ax.legend([std_poly], [r"$\pm 1\sigma$"], fontsize=9, frameon=True, loc="upper left", bbox_to_anchor=(0.0, 0.72))

        plt.tight_layout()
        plt.show()
    else:
        fig, axs = plt.subplots(3, 1, figsize=(10, 7.8), sharex=True)

        seed_lines0 = []
        seed_lines1 = []
        seed_lines2 = []

        for i, (s, d, tr) in enumerate(seed_rollouts):
            line0, = axs[0].plot(t_grid, vx_stack[i], alpha=0.25, label=f"seed {s}")
            line1, = axs[1].plot(t_grid, roll_stack[i], alpha=0.25, label=f"seed {s}")
            line2, = axs[2].plot(t_grid, pitch_stack[i], alpha=0.25, label=f"seed {s}")
            seed_lines0.append((s, line0))
            seed_lines1.append((s, line1))
            seed_lines2.append((s, line2))

        mean0, = axs[0].plot(t_grid, vx_mean, linewidth=2.5, label="mean")
        sm0, = axs[0].plot(t_grid, vx_mean_s, linewidth=2.5, label="smoothed mean")
        std0 = axs[0].fill_between(t_grid, vx_mean - vx_std, vx_mean + vx_std, alpha=0.2, label=r"$\pm 1\sigma$")
        av0 = axs[0].axhline(vx_avg, linestyle="--", linewidth=1.5, label=rf"avg = {vx_avg:.2f}")

        mean1, = axs[1].plot(t_grid, roll_mean, linewidth=2.5, label="mean")
        sm1, = axs[1].plot(t_grid, roll_mean_s, linewidth=2.5, label="smoothed mean")
        std1 = axs[1].fill_between(t_grid, roll_mean - roll_std, roll_mean + roll_std, alpha=0.2, label=r"$\pm 1\sigma$")
        av1 = axs[1].axhline(roll_avg, linestyle="--", linewidth=1.5, label=rf"avg = {roll_avg:.3f}")

        mean2, = axs[2].plot(t_grid, pitch_mean, linewidth=2.5, label="mean")
        sm2, = axs[2].plot(t_grid, pitch_mean_s, linewidth=2.5, label="smoothed mean")
        std2 = axs[2].fill_between(t_grid, pitch_mean - pitch_std, pitch_mean + pitch_std, alpha=0.2, label=r"$\pm 1\sigma$")
        av2 = axs[2].axhline(pitch_avg, linestyle="--", linewidth=1.5, label=rf"avg = {pitch_avg:.3f}")

        axs[0].set_ylabel(r"$v_x\,[\mathrm{m/s}]$")
        # axs[0].set_title(r"Base Forward Velocity")
        axs[0].grid(True, alpha=0.3)

        axs[1].set_ylabel(r"$\varphi\,[\mathrm{rad}]$")
        # axs[1].set_title(r"Base Roll")
        axs[1].grid(True, alpha=0.3)

        axs[2].set_ylabel(r"$\xi\,[\mathrm{rad}]$")
        axs[2].set_xlabel(r"$t\,[\mathrm{s}]$")
        # axs[2].set_title(r"Base Pitch")
        axs[2].grid(True, alpha=0.3)

        fig.align_ylabels(axs)

        def legend_with_std(ax, seed_lines, mean_line, smooth_line, avg_line, std_poly, loc_main, loc_std, anchor_std):
            seed_lines_sorted = [line for (_, line) in sorted(seed_lines, key=lambda x: x[0])]
            seed_labels_sorted = [f"seed {s}" for (s, _) in sorted(seed_lines, key=lambda x: x[0])]
            handles = seed_lines_sorted + [mean_line, smooth_line, avg_line]
            labels = seed_labels_sorted + ["mean", "smoothed mean", avg_line.get_label()]
            leg_main = ax.legend(handles, labels, ncol=3, fontsize=9, frameon=True, loc=loc_main)
            ax.add_artist(leg_main)
            ax.legend([std_poly], [r"$\pm 1\sigma$"], fontsize=9, frameon=True, loc=loc_std, bbox_to_anchor=anchor_std)

        legend_with_std(axs[0], seed_lines0, mean0, sm0, av0, std0, "upper left", "upper left", (0.0, 0.72))
        legend_with_std(axs[1], seed_lines1, mean1, sm1, av1, std1, "upper left", "upper left", (0.0, 0.72))
        legend_with_std(axs[2], seed_lines2, mean2, sm2, av2, std2, "upper left", "upper left", (0.0, 0.72))

        plt.tight_layout()
        plt.show()
