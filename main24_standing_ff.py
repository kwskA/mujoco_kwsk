# main_standing.py

import numpy as np
from pathlib import Path
from datetime import datetime

from mujoco import MjModel, MjData, mj_forward

from model.model_1024 import create_model_info

from control.control_methods.pd_ff_controller import PDFFController
from control.phases.standing_controller import StandingController

from simulation.objectives.objective_manager import ObjectiveManager
from simulation.objectives.time_objective import SimulationTimeObjective
from simulation.fall_detector import COMHeightFallDetector
from simulation.objectives.metabolic_energy_objective import MetabolicEnergyObjective
from simulation.objectives.muscle_registry import get_muscle_data_module
from simulation.objectives.com_path_length_objective import COMPathLengthObjective

from optim.initial_state_writer import InitialStateWriter
from optim.optimizer_factory import create_optimizer


ROOT_DIR = Path(__file__).parent

MODEL_INFO = create_model_info()
MUSCLES = MODEL_INFO.muscle_names

MUSCLE_DATA_MODULE = get_muscle_data_module(
    MODEL_INFO.name
)

# True  : 左右対称パラメータで最適化
# False : 全筋を個別パラメータで最適化
USE_SYMMETRIC_PARAMS = True

# センサ遅延 step数
# 2000 step = 10 s の場合，1 step = 5 ms
# 40 step = 200 ms
SENSOR_DELAY_STEPS = 40


def build_controller(model):
    """
    立位動作用の制御器を作成する。
    """

    return StandingController(
        name="Standing",
        control_method=PDFFController(
            model=model,
            muscles=MUSCLES,
            use_symmetric_params=USE_SYMMETRIC_PARAMS,
            symmetric_muscle_pairs=MODEL_INFO.symmetric_muscle_pairs,
            sensor_delay_steps=SENSOR_DELAY_STEPS,
        ),
    )

def build_objective_manager():
    return ObjectiveManager([
        SimulationTimeObjective(
            target_steps=1000,
            weight=1000.0,
            apply_to="all",
        ),
        MetabolicEnergyObjective(
            muscle_names=MUSCLES,
            muscle_data_module=MUSCLE_DATA_MODULE,
            weight=0.001,
        ),
        COMPathLengthObjective(
            weight=1,
            axes=(0, 1, 2),
            apply_to="all",
        ),],
        total_cost_mode = "sum",
    )


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

    com_init_height = float(data.subtree_com[0][2])

    return COMHeightFallDetector(
        com_init_height=com_init_height,
        threshold_ratio=0.9,
    )

def get_base_name(name):
    if name.endswith("_r"):
        return name[:-2]
    if name.endswith("_l"):
        return name[:-2]
    return name

def main():

    sim_steps = 1000

    sigma0 = 0.2
    popsize = 100
    maxiter = 100000

    n_jobs = 60
    reserve_cores = 1

    checkpoint_interval = 5000

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    symmetry_tag = "symmetric" if USE_SYMMETRIC_PARAMS else "individual"

    result_dir = (
        ROOT_DIR
        / "results"
        / f"{timestamp}_standing_{MODEL_INFO.name}_{symmetry_tag}"
    )

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

    controller = build_controller(model)

    x0 = controller.make_initial_params(
        init_Kp=10.0,
        init_Kd=2.0,
        init_target_length=init_len,
    )

    print("[main_standing]")
    print("model =", MODEL_INFO.name)
    print("num_muscles =", len(MUSCLES))
    print("use_symmetric_params =", USE_SYMMETRIC_PARAMS)
    print("param_dim =", x0.size)
    # print("expected_param_dim =", controller.get_expected_param_dim())

    print("x0 shape =", x0.shape)
    # print("Kp =", kp_array)
    # print("Kd =", kd_array)

    initial_state_dir = result_dir / "initial_state"

    InitialStateWriter(
        save_dir=str(initial_state_dir),
        model=model,
        data=data,
        muscle_names=MUSCLES,
        model_path=MODEL_INFO.model_path,
        extra_info={
            "sim_steps": sim_steps,
            "controller": "StandingController + PDFFController",
            "use_symmetric_params": USE_SYMMETRIC_PARAMS,
            "sensor_delay_steps": SENSOR_DELAY_STEPS,
            "num_muscles": len(MUSCLES),
            "param_dim": int(x0.size),
            "init_Kp": 10.0,
            "init_Kd": 2.0,
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
    print("========== Standing optimization finished ==========")
    print("best_cost =", result["best_cost"])
    print("result_dir =", result_dir)


if __name__ == "__main__":
    main()