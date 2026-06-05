# main_standing.py

import numpy as np
from pathlib import Path
from datetime import datetime

from mujoco import MjModel, MjData, mj_forward

from model.model_1018 import create_model_info

from control.control_methods.pd_controller import PDController
from control.phases.standing_controller import StandingController

from simulation.objectives.objective_manager import ObjectiveManager
from simulation.objectives.time_objective import SimulationTimeObjective
from simulation.fall_detector import COMHeightFallDetector

from optim.initial_state_writer import InitialStateWriter
from optim.optimizer_factory import create_optimizer


ROOT_DIR = Path(__file__).parent

MODEL_INFO = create_model_info()
MUSCLES = MODEL_INFO.muscle_names


def build_controller(model):
    """
    立位動作用の制御器を作成する。
    """

    return StandingController(
        name="Standing",
        control_method=PDController(
            model=model,
            muscles=MUSCLES,
        ),
    )


def build_objective_manager():
    """
    立位動作用の評価関数群を作成する。
    """

    return ObjectiveManager([
        SimulationTimeObjective(
            target_steps=1000,
            weight=20.0,
            apply_to="all",
        ),
    ])


def build_fall_detector(model):
    """
    XML初期姿勢からCOM高さを計算し，
    転倒判定器を作成する。
    """

    data = MjData(model)

    data.qpos[:] = model.key_qpos[0].copy()
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0

    mj_forward(model, data)

    com_init_height = float(
        data.subtree_com[0][2]
    )

    return COMHeightFallDetector(
        com_init_height=com_init_height,
        threshold_ratio=0.9,
    )


def main():

    sim_steps = 1000

    sigma0 = 0.5
    popsize = 25
    maxiter = 200

    n_jobs = 6
    reserve_cores = 1

    checkpoint_interval = 100

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_dir = ROOT_DIR / "results" / f"standing_{MODEL_INFO.name}_{timestamp}"

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

    init_len = np.array([
        data.ten_length[t_id]
        for t_id in tendon_ids
    ])

    initial_state_dir = result_dir / "initial_state"

    InitialStateWriter(
        save_dir=str(initial_state_dir),
        model=model,
        data=data,
        muscle_names=MUSCLES,
        model_path=MODEL_INFO.model_path,
        extra_info={
            "sim_steps": sim_steps,
            "controller": "StandingController + PDController",
            "init_Kp": 10.0,
            "init_Kd": 2.0,
        },
    ).write_all()

    controller = build_controller(model)

    x0 = controller.make_initial_params(
        init_Kp=10.0,
        init_Kd=2.0,
        init_target_length=init_len,
    )

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
    print("========== Standing optimization finished ==========")
    print("best_cost =", result["best_cost"])
    print("result_dir =", result_dir)


if __name__ == "__main__":
    main()