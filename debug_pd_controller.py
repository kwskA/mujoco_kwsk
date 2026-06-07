# debug_pd_controller.py

import json
import numpy as np
from pathlib import Path
from datetime import datetime

from mujoco import MjModel, MjData, mj_forward

from model.model_1018 import create_model_info

from control.control_methods.pd_controller import PDController
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

DEBUG_KP = 5.0
DEBUG_KD = 2.0

# target_length = initial_length * TARGET_SCALE
# 1.0なら初期筋長をそのまま目標筋長にする
TARGET_SCALE = 1.0

USE_SYMMETRIC_PARAMS = False

OUTPUT_VIDEOS = True
OUTPUT_CSV = True


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


def main():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_dir = (
        ROOT_DIR
        / "debug_results"
        / f"debug_pd_{MODEL_INFO.name}_{timestamp}"
    )

    result_dir.mkdir(parents=True, exist_ok=True)

    model = MjModel.from_xml_path(MODEL_INFO.model_path)
    data = MjData(model)

    data.qpos[:] = model.key_qpos[0].copy()
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0

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

    initial_length = np.array([
        data.ten_length[t_id]
        for t_id in tendon_ids
    ])

    target_length = initial_length * TARGET_SCALE

    controller = build_controller(model)

    params = controller.make_initial_params(
        init_Kp=DEBUG_KP,
        init_Kd=DEBUG_KD,
        init_target_length=target_length,
    )

    controller.set_params_from_vector(params)

    np.save(
        result_dir / "debug_params.npy",
        params,
    )

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
                save_path=str(video_dir / f"debug_pd_{camera_name}.mp4"),
                tendon_ids=tendon_ids,
                actuator_ids=actuator_ids,
                color_tendons=True,
            )

    debug_info = {
        "model": MODEL_INFO.name,
        "model_path": MODEL_INFO.model_path,
        "sim_steps": SIM_STEPS,
        "debug_Kp": DEBUG_KP,
        "debug_Kd": DEBUG_KD,
        "target_scale": TARGET_SCALE,
        "use_symmetric_params": USE_SYMMETRIC_PARAMS,
        "param_dim": int(params.size),
        "num_muscles": len(MUSCLES),
        "fallen": bool(result["fallen"]),
        "survived_steps": int(result["survived_steps"]),
        "total_cost": float(total_cost),
        "details": details,
    }

    with open(result_dir / "debug_info.json", "w", encoding="utf-8") as f:
        json.dump(debug_info, f, indent=4, ensure_ascii=False)

    print()
    print("========== Debug PD simulation finished ==========")
    print("result_dir =", result_dir)
    print("total_cost =", total_cost)
    print("details =", details)
    print("fallen =", result["fallen"])
    print("survived_steps =", result["survived_steps"])


if __name__ == "__main__":
    main()