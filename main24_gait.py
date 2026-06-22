# main_gait.py

import numpy as np
from pathlib import Path
from datetime import datetime

from mujoco import MjModel, MjData, mj_forward

from model.model_1024 import create_model_info

from control.control_methods.pd_controller import PDController
from control.phases.gait_controller import GaitController
from control.initial_pose.gait_pose import set_gait_pose

from simulation.objectives.objective_manager import ObjectiveManager
from simulation.objectives.time_objective import SimulationTimeObjective
from simulation.objectives.metabolic_energy_objective import MetabolicEnergyObjective
from simulation.objectives.walking_speed_objective import WalkingSpeedObjective
from simulation.objectives.muscle_registry import get_muscle_data_module
from simulation.fall_detector import COMHeightFallDetector

from optim.initial_state_writer import InitialStateWriter
from optim.optimizer_factory import create_optimizer


ROOT_DIR = Path(__file__).parent

MODEL_INFO = create_model_info()
MUSCLES = MODEL_INFO.muscle_names

MUSCLE_DATA_MODULE = get_muscle_data_module(MODEL_INFO.name)

USE_SYMMETRIC_PARAMS = True


def build_controller(model):
    """
    歩行動作用の制御器を作成する。
    """

    return GaitController(
        name="Gait",
        model=model,
        control_method=PDController(
            model=model,
            muscles=MUSCLES,
            use_symmetric_params=USE_SYMMETRIC_PARAMS,
            symmetric_muscle_pairs=MODEL_INFO.symmetric_muscle_pairs,
            use_gait_states=True,
            gait_states=[
                "EarlyStance",
                "LateStance",
                "Liftoff",
                "Swing",
                "Landing",
            ],
        ),
    )


def build_objective_manager():
    return ObjectiveManager([
        SimulationTimeObjective(
            target_steps=2000,
            weight=20.0,
            apply_to="all",
        ),
        MetabolicEnergyObjective(
            muscle_names=MUSCLES,
            muscle_data_module=MUSCLE_DATA_MODULE,
            weight=0.001,
        ),
        WalkingSpeedObjective(
            min_velocity=1.0,
            weight=10.0,
            apply_to="all",
            qpos_index=0,
        ),
    ])


def build_fall_detector(model):
    data = MjData(model)

    data.qpos[:] = model.key_qpos[0].copy()

    set_gait_pose(
        model=model,
        data=data,
        pose_name="EarlyStance",
    )

    com_init_height = float(data.subtree_com[0][2])

    return COMHeightFallDetector(
        com_init_height=com_init_height,
        threshold_ratio=0.9,
    )


def main():

    sim_steps = 2000

    sigma0 = 0.2
    popsize = 100
    maxiter = 500

    n_jobs = 6
    reserve_cores = 1

    checkpoint_interval = 200

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    symmetry_tag = "symmetric" if USE_SYMMETRIC_PARAMS else "individual"

    result_dir = (
        ROOT_DIR
        / "results"
        / f"{timestamp}_gait_{MODEL_INFO.name}_{symmetry_tag}"
    )

    model = MjModel.from_xml_path(MODEL_INFO.model_path)
    data = MjData(model)

    data.qpos[:] = model.key_qpos[0].copy()

    set_gait_pose(
        model=model,
        data=data,
        pose_name="EarlyStance",
    )

    initial_qpos = data.qpos.copy()

    tendon_ids = np.array([
        model.tendon(f"{m}_tendon").id
        for m in MUSCLES
    ], dtype=int)

    init_len = np.array([
        data.ten_length[t_id]
        for t_id in tendon_ids
    ])

    controller = build_controller(model)

    x0 = controller.make_initial_params(
        init_Kp=10.0,
        init_Kd=2.0,
        init_target_length=init_len,
    )

    print("[main_gait]")
    print("model =", MODEL_INFO.name)
    print("num_muscles =", len(MUSCLES))
    print("use_symmetric_params =", USE_SYMMETRIC_PARAMS)
    print("param_dim =", x0.size)
    print("x0 shape =", x0.shape)

    initial_state_dir = result_dir / "initial_state"

    InitialStateWriter(
        save_dir=str(initial_state_dir),
        model=model,
        data=data,
        muscle_names=MUSCLES,
        model_path=MODEL_INFO.model_path,
        extra_info={
            "sim_steps": sim_steps,
            "controller": "GaitController + PDController",
            "use_symmetric_params": USE_SYMMETRIC_PARAMS,
            "num_muscles": len(MUSCLES),
            "param_dim": int(x0.size),
            "init_Kp": 10.0,
            "init_Kd": 2.0,
            "gait_states": [
                "EarlyStance",
                "LateStance",
                "Liftoff",
                "Swing",
                "Landing",
            ],
        },
    ).write_all()

    optimizer = create_optimizer(
        optimizer_type="cmaes",

        model_path=MODEL_INFO.model_path,
        controller=controller,
        controller_builder=build_controller,
        objective_manager_builder=build_objective_manager,

        sim_steps=sim_steps,
        result_dir=str(result_dir),

        initial_qpos=initial_qpos,
        muscle_names=MUSCLES,

        sigma0=sigma0,
        popsize=popsize,
        maxiter=maxiter,

        n_jobs=n_jobs,
        reserve_cores=reserve_cores,

        fall_detector_builder=build_fall_detector,

        checkpoint_interval=checkpoint_interval,
    )

    result = optimizer.optimize(x0)

    print()
    print("========== Gait optimization finished ==========")
    print("best_cost =", result["best_cost"])
    print("result_dir =", result_dir)


if __name__ == "__main__":
    main()