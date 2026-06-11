# debug_pd_energy.py

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
from simulation.objectives.metabolic_energy_objective import MetabolicEnergyObjective
from simulation.objectives.muscle_registry import get_muscle_data_module
from simulation.fall_detector import COMHeightFallDetector

from render.video_renderer import VideoRenderer


ROOT_DIR = Path(__file__).parent

MODEL_INFO = create_model_info()
MUSCLES = MODEL_INFO.muscle_names
MUSCLE_DATA_MODULE = get_muscle_data_module(MODEL_INFO.name)


SIM_STEPS = 1000
USE_SYMMETRIC_PARAMS = False

OUTPUT_VIDEOS = True
OUTPUT_CSV = True


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


def build_objective_manager():
    return ObjectiveManager([
        SimulationTimeObjective(
            target_steps=SIM_STEPS,
            weight=20.0,
            apply_to="all",
        ),
        MetabolicEnergyObjective(
            muscle_names=MUSCLES,
            muscle_data_module=MUSCLE_DATA_MODULE,
            weight=0.001,
        ),
    ])


def build_fall_detector(model, initial_qpos):
    data = MjData(model)
    data.qpos[:] = initial_qpos.copy()
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0

    mj_forward(model, data)

    com_init_height = float(data.subtree_com[0][1])

    return COMHeightFallDetector(
        com_init_height=com_init_height,
        threshold_ratio=0.9,
    )


def get_joint_qpos_dim(model, joint_id):
    joint_type = model.jnt_type[joint_id]

    if joint_type == 0:
        return 7

    if joint_type == 1:
        return 4

    return 1


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


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_dir = (
        ROOT_DIR
        / "debug_results"
        / f"debug_pd_energy_{MODEL_INFO.name}_{timestamp}"
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

    kp_array, kd_array, target_length = build_params_from_debug_dict(MUSCLES)

    controller = build_controller(model)

    params = np.concatenate([
        kp_array,
        kd_array,
        target_length,
    ])

    controller.set_params_from_vector(params)

    np.save(result_dir / "debug_params.npy", params)

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
                save_path=str(video_dir / f"debug_pd_{camera_name}.mp4"),
                tendon_ids=tendon_ids,
                actuator_ids=actuator_ids,
                color_tendons=True,
            )

    debug_info = {
        "model": MODEL_INFO.name,
        "model_path": MODEL_INFO.model_path,
        "sim_steps": SIM_STEPS,
        "use_symmetric_params": USE_SYMMETRIC_PARAMS,
        "param_dim": int(params.size),
        "num_muscles": len(MUSCLES),
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
        "fallen": bool(result["fallen"]),
        "survived_steps": int(result["survived_steps"]),
        "total_cost": float(total_cost),
        "details": details,
        "metabolic": {
            "mean_power": float(details.get("mean_metabolic_power", 0.0)),
            "energy_joule": float(details.get("metabolic_energy_joule", 0.0)),
            "cost": float(details.get("metabolic_energy", 0.0)),
        },
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

    print()
    print("========== Metabolic Energy ==========")
    print("mean_power [W] =", details.get("mean_metabolic_power", None))
    print("energy [J] =", details.get("metabolic_energy_joule", None))
    print("metabolic_cost =", details.get("metabolic_energy", None))

    print()
    print("========== Applied Debug Parameters ==========")
    for i, muscle in enumerate(MUSCLES):
        print(
            f"{muscle:15s} "
            f"Kp={kp_array[i]:.6f} "
            f"Kd={kd_array[i]:.6f} "
            f"target_length={target_length[i]:.6f}"
        )


if __name__ == "__main__":
    main()