# debug_pd_com.py

import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mujoco import MjModel, MjData, mj_forward, mj_step

from model.model_1018 import create_model_info

from control.control_methods.pd_controller import PDController
from control.phases.standing_controller import StandingController

from render.video_renderer import VideoRenderer


ROOT_DIR = Path(__file__).parent

MODEL_INFO = create_model_info()
MUSCLES = MODEL_INFO.muscle_names


SIM_STEPS = 1000
USE_SYMMETRIC_PARAMS = False

OUTPUT_VIDEOS = True


DEBUG_PARAMS = {
    "hamstrings_r": {"kp": 4.703437051343508, "kd": 2.166304984937728, "target_length": 1.9448615943341514},
    "bifemsh_r": {"kp": 6.424473775155979, "kd": 0.36920485643695955, "target_length": 0.7469972371752154},
    "glut_max_r": {"kp": 3.731266043280769, "kd": 8.823220069382815, "target_length": 0.21529054183644347},
    "iliopsoas_r": {"kp": 9.440598266874199, "kd": 3.5281321434813524, "target_length": 0.8951164643775043},
    "rect_fem_r": {"kp": 8.316163342636795, "kd": 1.3993094237669734, "target_length": 0.016905067915034246},
    "vasti_r": {"kp": 4.925765153207904, "kd": 9.889890526003327, "target_length": 0.00013127256008945996},
    "gastroc_r": {"kp": 7.312496733979348, "kd": 0.007878463212862384, "target_length": 0.5542531580042915},
    "soleus_r": {"kp": 8.931651616487965, "kd": 0.4773773120211262, "target_length": 1.2196888556763283},
    "tib_ant_r": {"kp": 7.793636889175427, "kd": 0.2997567549999156, "target_length": 1.988009795588802},

    "hamstrings_l": {"kp": 4.703437051343508, "kd": 2.166304984937728, "target_length": 1.9448615943341514},
    "bifemsh_l": {"kp": 6.424473775155979, "kd": 0.36920485643695955, "target_length": 0.7469972371752154},
    "glut_max_l": {"kp": 3.731266043280769, "kd": 8.823220069382815, "target_length": 0.21529054183644347},
    "iliopsoas_l": {"kp": 9.440598266874199, "kd": 3.5281321434813524, "target_length": 0.8951164643775043},
    "rect_fem_l": {"kp": 8.316163342636795, "kd": 1.3993094237669734, "target_length": 0.016905067915034246},
    "vasti_l": {"kp": 4.925765153207904, "kd": 9.889890526003327, "target_length": 0.00013127256008945996},
    "gastroc_l": {"kp": 7.312496733979348, "kd": 0.007878463212862384, "target_length": 0.5542531580042915},
    "soleus_l": {"kp": 8.931651616487965, "kd": 0.4773773120211262, "target_length": 1.2196888556763283},
    "tib_ant_l": {"kp": 7.793636889175427, "kd": 0.2997567549999156, "target_length": 1.988009795588802},
}


def build_params_from_debug_dict(muscles):
    kp = []
    kd = []
    target_length = []

    for muscle in muscles:
        if muscle not in DEBUG_PARAMS:
            raise ValueError(f"DEBUG_PARAMS not found for muscle: {muscle}")

        kp.append(DEBUG_PARAMS[muscle]["kp"])
        kd.append(DEBUG_PARAMS[muscle]["kd"])
        target_length.append(DEBUG_PARAMS[muscle]["target_length"])

    return (
        np.array(kp, dtype=float),
        np.array(kd, dtype=float),
        np.array(target_length, dtype=float),
    )


def build_controller(model):
    return StandingController(
        name="Standing",
        control_method=PDController(
            model=model,
            muscles=MUSCLES,
            use_symmetric_params=USE_SYMMETRIC_PARAMS,
            symmetric_muscle_pairs=MODEL_INFO.symmetric_muscle_pairs,
        ),
    )


def save_com_csv(save_path, time, com_history):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "com_x", "com_y", "com_z"])

        for t, com in zip(time, com_history):
            writer.writerow([
                float(t),
                float(com[0]),
                float(com[1]),
                float(com[2]),
            ])


def plot_com_components(time, com_history, save_path):
    plt.figure(figsize=(10, 5))

    plt.plot(time, com_history[:, 0], label="COM X")
    plt.plot(time, com_history[:, 1], label="COM Y")
    plt.plot(time, com_history[:, 2], label="COM Z")

    plt.xlabel("Time [s]")
    plt.ylabel("COM position [m]")
    plt.title("COM components over time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_com_trajectory_mujoco_axes(com_history, save_path):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        com_history[:, 0],
        com_history[:, 1],
        com_history[:, 2],
        linewidth=2,
        label="COM trajectory",
    )

    ax.scatter(
        com_history[0, 0],
        com_history[0, 1],
        com_history[0, 2],
        s=50,
        label="Start",
    )

    ax.scatter(
        com_history[-1, 0],
        com_history[-1, 1],
        com_history[-1, 2],
        s=50,
        label="End",
    )

    ax.set_xlabel("MuJoCo X")
    ax.set_ylabel("MuJoCo Y")
    ax.set_zlabel("MuJoCo Z")
    ax.set_title("COM trajectory in MuJoCo axes")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_com_trajectory_assumed_axes(com_history, save_path):
    """
    仮定：
    X = 進行方向
    Y = 垂直方向
    Z = 左右方向

    そのまま X, Y, Z として3D描画する。
    """

    forward = com_history[:, 0]
    vertical = com_history[:, 1]
    lateral = com_history[:, 2]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        forward,
        lateral,
        vertical,
        linewidth=2,
        label="COM trajectory",
    )

    ax.scatter(
        forward[0],
        lateral[0],
        vertical[0],
        s=50,
        label="Start",
    )

    ax.scatter(
        forward[-1],
        lateral[-1],
        vertical[-1],
        s=50,
        label="End",
    )

    ax.set_xlabel("Forward direction: X [m]")
    ax.set_ylabel("Lateral direction: Z [m]")
    ax.set_zlabel("Vertical direction: Y [m]")
    ax.set_title("COM trajectory assuming Y is vertical")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compute_total_com_path_length(com_history):
    diffs = np.diff(com_history, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return float(np.sum(segment_lengths))


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_dir = (
        ROOT_DIR
        / "debug_results"
        / f"debug_com_{MODEL_INFO.name}_{timestamp}"
    )

    result_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    model = MjModel.from_xml_path(MODEL_INFO.model_path)
    data = MjData(model)

    data.qpos[:] = model.key_qpos[0].copy()
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0

    mj_forward(model, data)

    initial_qpos = data.qpos.copy()

    kp_array, kd_array, target_length = build_params_from_debug_dict(MUSCLES)

    controller = build_controller(model)

    params = np.concatenate([
        kp_array,
        kd_array,
        target_length,
    ])

    controller.set_params_from_vector(params)

    np.save(result_dir / "debug_params.npy", params)

    time_history = []
    com_history = []

    dt = float(model.opt.timestep)

    for step in range(SIM_STEPS):
        t = step * dt

        com = data.subtree_com[0].copy()

        time_history.append(t)
        com_history.append(com)

        ctrl = controller.compute_ctrl(data)
        data.ctrl[:] = ctrl

        mj_step(model, data)

    time_history = np.asarray(time_history, dtype=float)
    com_history = np.asarray(com_history, dtype=float)

    total_com_path_length = compute_total_com_path_length(com_history)

    com_min = np.min(com_history, axis=0)
    com_max = np.max(com_history, axis=0)
    com_range = com_max - com_min

    save_com_csv(
        save_path=result_dir / "com_history.csv",
        time=time_history,
        com_history=com_history,
    )

    plot_com_components(
        time=time_history,
        com_history=com_history,
        save_path=plot_dir / "com_components_time.png",
    )

    plot_com_trajectory_mujoco_axes(
        com_history=com_history,
        save_path=plot_dir / "com_trajectory_mujoco_xyz.png",
    )

    plot_com_trajectory_assumed_axes(
        com_history=com_history,
        save_path=plot_dir / "com_trajectory_assumed_y_vertical.png",
    )

    if OUTPUT_VIDEOS:
        video_dir = result_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        camera_settings = {
            "front": {"azimuth": 90.0, "elevation": -10.0, "distance": 3.0},
            "side": {"azimuth": 0.0, "elevation": -10.0, "distance": 3.0},
            "diagonal": {"azimuth": 45.0, "elevation": -15.0, "distance": 3.0},
            "oblique": {"azimuth": 135.0, "elevation": -20.0, "distance": 3.2},
        }

        for camera_name, camera_param in camera_settings.items():
            video_renderer = VideoRenderer(
                model=model,
                controller=controller,
                sim_steps=SIM_STEPS,
                initial_qpos=initial_qpos,
                tracking_body_name=MODEL_INFO.tracking_body_name,
                camera_distance=camera_param["distance"],
                camera_azimuth=camera_param["azimuth"],
                camera_elevation=camera_param["elevation"],
            )

            video_renderer.render(
                save_path=str(video_dir / f"debug_com_{camera_name}.mp4"),
                tendon_ids=None,
                actuator_ids=None,
                color_tendons=False,
            )

    debug_info = {
        "model": MODEL_INFO.name,
        "model_path": MODEL_INFO.model_path,
        "sim_steps": SIM_STEPS,
        "dt": dt,
        "use_symmetric_params": USE_SYMMETRIC_PARAMS,
        "param_dim": int(params.size),
        "num_muscles": len(MUSCLES),
        "total_com_path_length": total_com_path_length,
        "com_min": {
            "x": float(com_min[0]),
            "y": float(com_min[1]),
            "z": float(com_min[2]),
        },
        "com_max": {
            "x": float(com_max[0]),
            "y": float(com_max[1]),
            "z": float(com_max[2]),
        },
        "com_range": {
            "x": float(com_range[0]),
            "y": float(com_range[1]),
            "z": float(com_range[2]),
        },
        "axis_assumption": {
            "forward": "x",
            "vertical": "y",
            "lateral": "z",
        },
        "kp": {
            muscle: float(kp_array[i])
            for i, muscle in enumerate(MUSCLES)
        },
        "kd": {
            muscle: float(kd_array[i])
            for i, muscle in enumerate(MUSCLES)
        },
        "target_length": {
            muscle: float(target_length[i])
            for i, muscle in enumerate(MUSCLES)
        },
    }

    with open(result_dir / "debug_com_info.json", "w", encoding="utf-8") as f:
        json.dump(debug_info, f, indent=4, ensure_ascii=False)

    print()
    print("========== Debug COM trajectory finished ==========")
    print("result_dir =", result_dir)
    print("total_com_path_length =", total_com_path_length)
    print()
    print("COM min =", com_min)
    print("COM max =", com_max)
    print("COM range =", com_range)
    print()
    print("Axis assumption:")
    print("  X = forward")
    print("  Y = vertical")
    print("  Z = lateral")
    print()
    print("wrote:", result_dir / "com_history.csv")
    print("wrote:", plot_dir / "com_components_time.png")
    print("wrote:", plot_dir / "com_trajectory_mujoco_xyz.png")
    print("wrote:", plot_dir / "com_trajectory_assumed_y_vertical.png")


if __name__ == "__main__":
    main()