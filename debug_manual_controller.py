import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mujoco import MjModel, MjData, mj_forward

from model.model_1018 import create_model_info

from control.control_methods.debug_activation_controller import DebugActivationController
from control.phases.standing_controller import StandingController

from simulation.runner import SimulationRunner
from simulation.logs.csv_log_writer import CSVLogWriter
from simulation.objectives.objective_manager import ObjectiveManager
from simulation.objectives.time_objective import SimulationTimeObjective
from simulation.fall_detector import COMHeightFallDetector

from render.video_renderer import VideoRenderer


ROOT_DIR = Path(__file__).parent

MODEL_INFO = create_model_info()
MUSCLES = MODEL_INFO.muscle_names


# =========================
# デバッグ設定
# =========================

SIM_STEPS = 1000

# 指定がない筋はこの値になる
DEFAULT_ACTIVATION = 0

# 筋ごとにactivationを指定
# 最初は全筋1.0なので空でOK
# 例:
# ACTIVATION_DICT = {
#     "hamstrings_r": 1.0,
#     "hamstrings_l": 1.0,
#     "tib_ant_r": 0.0,
#     "tib_ant_l": 0.0,
# }
ACTIVATION_DICT = {}

OUTPUT_VIDEOS = True
OUTPUT_CSV = True
OUTPUT_PLOTS = True


def build_controller(model):
    return StandingController(
        name="Standing",
        control_method=DebugActivationController(
            model=model,
            muscles=MUSCLES,
            activation_dict=ACTIVATION_DICT,
            default_activation=DEFAULT_ACTIVATION,
        ),
    )


def build_objective_manager():
    return ObjectiveManager([
        SimulationTimeObjective(
            target_steps=SIM_STEPS,
            weight=20.0,
            apply_to="all",
        ),
    ])


def build_fall_detector(model, initial_qpos):
    data = MjData(model)

    data.qpos[:] = initial_qpos.copy()
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.ctrl[:] = 0.0

    mj_forward(model, data)

    com_init_height = float(data.subtree_com[0][2])

    return COMHeightFallDetector(
        com_init_height=com_init_height,
        threshold_ratio=0.9,
    )


def get_qpos_names(model):
    qpos_names = [None for _ in range(model.nq)]

    for joint_id in range(model.njnt):
        joint_name = model.joint(joint_id).name
        qpos_adr = model.jnt_qposadr[joint_id]
        qpos_dim = get_joint_qpos_dim(model, joint_id)

        if qpos_dim == 1:
            qpos_names[qpos_adr] = joint_name
        else:
            for k in range(qpos_dim):
                qpos_names[qpos_adr + k] = f"{joint_name}_{k}"

    for i, name in enumerate(qpos_names):
        if name is None:
            qpos_names[i] = f"qpos_{i}"

    return qpos_names


def get_joint_qpos_dim(model, joint_id):
    joint_type = model.jnt_type[joint_id]

    # free joint
    if joint_type == 0:
        return 7

    # ball joint
    if joint_type == 1:
        return 4

    # slide / hinge joint
    return 1


def plot_muscle_length_activation(
    muscles_csv_path,
    save_dir,
    muscle_names,
):
    """
    各筋について，
    実際の筋長・activationを1枚のグラフに保存する。
    """

    save_dir.mkdir(parents=True, exist_ok=True)

    with open(muscles_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        print("[debug plot] muscles.csv is empty.")
        return

    time = np.array([
        float(row["time"])
        for row in rows
    ])

    for muscle in muscle_names:

        length_key = f"{muscle}.tendon_length"
        activation_key = f"{muscle}.activation"

        if length_key not in rows[0]:
            print(f"[debug plot] skipped {muscle}: {length_key} not found")
            continue

        if activation_key not in rows[0]:
            print(f"[debug plot] skipped {muscle}: {activation_key} not found")
            continue

        muscle_length = np.array([
            float(row[length_key])
            for row in rows
        ])

        activation = np.array([
            float(row[activation_key])
            for row in rows
        ])

        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(
            time,
            muscle_length,
            color="tab:blue",
            linewidth=2,
            label="tendon_length",
        )

        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel(
            "Muscle length [m]",
            color="tab:blue",
        )
        ax1.tick_params(
            axis="y",
            colors="tab:blue",
        )
        ax1.grid(True)

        ax2 = ax1.twinx()

        ax2.plot(
            time,
            activation,
            color="tab:green",
            linewidth=2,
            label="activation",
        )

        ax2.set_ylabel(
            "Activation",
            color="tab:green",
        )
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(
            axis="y",
            colors="tab:green",
        )

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper right",
        )

        plt.title(muscle)
        plt.tight_layout()

        save_path = save_dir / f"{muscle}.png"
        plt.savefig(save_path)
        plt.close()

        print(f"[debug plot] wrote {save_path}")


def main():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_dir = (
        ROOT_DIR
        / "debug_results"
        / f"debug_activation_{MODEL_INFO.name}_{timestamp}"
    )

    result_dir.mkdir(parents=True, exist_ok=True)

    model = MjModel.from_xml_path(MODEL_INFO.model_path)
    data = MjData(model)

    data.qpos[:] = model.key_qpos[0].copy()
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.ctrl[:] = 0.0

    mj_forward(model, data)

    initial_qpos = data.qpos.copy()

    tendon_ids = np.array([
        model.tendon(f"{m}_tendon").id
        for m in MUSCLES
    ], dtype=int)

    actuator_ids = np.array([
        model.actuator(m).id
        for m in MUSCLES
    ], dtype=int)

    controller = build_controller(model)

    objective_manager = build_objective_manager()

    fall_detector = build_fall_detector(
        model=model,
        initial_qpos=initial_qpos,
    )

    qpos_names = get_qpos_names(model)

    qvel_names = [
        f"dof_{i}"
        for i in range(model.nv)
    ]

    runner = SimulationRunner(
        model=model,
        controller=controller,
        objective_manager=objective_manager,
        sim_steps=SIM_STEPS,
        initial_qpos=initial_qpos,
        fall_detector=fall_detector,
        qpos_names=qpos_names,
        qvel_names=qvel_names,
        muscle_names=MUSCLES,
        tendon_ids=tendon_ids,
        actuator_ids=actuator_ids,
        enable_log=True,
    )

    result = runner.run(data)
    sim_log = result["simulation_log"]

    total_cost, details = objective_manager.finalize()

    if OUTPUT_CSV:
        csv_dir = result_dir / "csv"
        CSVLogWriter(str(csv_dir)).write_all(sim_log)

        if OUTPUT_PLOTS:
            plot_dir = result_dir / "plots" / "muscle_length_activation"

            plot_muscle_length_activation(
                muscles_csv_path=csv_dir / "muscles.csv",
                save_dir=plot_dir,
                muscle_names=MUSCLES,
            )

    if OUTPUT_VIDEOS:
        video_dir = result_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        camera_settings = {
            "front": {
                "azimuth": 90.0,
                "elevation": -10.0,
                "distance": 3.0,
            },
            "side": {
                "azimuth": 0.0,
                "elevation": -10.0,
                "distance": 3.0,
            },
            "diagonal": {
                "azimuth": 45.0,
                "elevation": -15.0,
                "distance": 3.0,
            },
            "oblique": {
                "azimuth": 135.0,
                "elevation": -20.0,
                "distance": 3.2,
            },
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
                save_path=str(video_dir / f"debug_activation_{camera_name}.mp4"),
                tendon_ids=tendon_ids,
                actuator_ids=actuator_ids,
                color_tendons=True,
            )

    activation_values = {
        muscle: float(
            ACTIVATION_DICT.get(
                muscle,
                DEFAULT_ACTIVATION,
            )
        )
        for muscle in MUSCLES
    }

    debug_info = {
        "model": MODEL_INFO.name,
        "model_path": MODEL_INFO.model_path,
        "sim_steps": SIM_STEPS,
        "controller": "DebugActivationController",
        "default_activation": DEFAULT_ACTIVATION,
        "activation_values": activation_values,
        "num_muscles": len(MUSCLES),
        "fallen": bool(result["fallen"]),
        "survived_steps": int(result["survived_steps"]),
        "total_cost": float(total_cost),
        "details": details,
    }

    with open(result_dir / "debug_info.json", "w", encoding="utf-8") as f:
        json.dump(debug_info, f, indent=4, ensure_ascii=False)

    print()
    print("========== Debug activation simulation finished ==========")
    print("result_dir =", result_dir)
    print("total_cost =", total_cost)
    print("details =", details)
    print("fallen =", result["fallen"])
    print("survived_steps =", result["survived_steps"])


if __name__ == "__main__":
    main()