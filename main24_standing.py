# main_standing.py

import numpy as np
from pathlib import Path
from datetime import datetime

from mujoco import MjModel, MjData, mj_forward

from model.model_1024 import create_model_info

from control.control_methods.pd_controller import PDController
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


def build_controller(model):
    """
    立位動作用の制御器を作成する。
    """

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
            target_steps=1000,
            weight=20.0,
            apply_to="all",
        ),
        MetabolicEnergyObjective(
            muscle_names=MUSCLES,
            muscle_data_module=MUSCLE_DATA_MODULE,
            weight=0.001,
        ),
        COMPathLengthObjective(
            weight=1.0,
            axes=(0, 1, 2),
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
    popsize = 50
    maxiter = 1000

    n_jobs = 6
    reserve_cores = 1

    checkpoint_interval = 500

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    symmetry_tag = "symmetric" if USE_SYMMETRIC_PARAMS else "individual"

    result_dir = (
        ROOT_DIR
        / "results"
        / f"standing_{MODEL_INFO.name}_{timestamp}_{symmetry_tag}"
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

    # SCONE_KP_KD = {
    #     "hamstrings": {"Kp": 11.447057788827806, "Kd": 0.5672356839786517},
    #     "bifemsh":   {"Kp": 10.736705385807000, "Kd": 0.9883024640312215},
    #     "glut_max":  {"Kp": 8.379833415780013,  "Kd": 8.762719053298497},
    #     "iliopsoas": {"Kp": 8.510325115799787,  "Kd": 4.820861976809738},
    #     "rect_fem":  {"Kp": 12.243241227413819, "Kd": 0.35661551701414285},
    #     "vasti":     {"Kp": 13.020649537404097, "Kd": 5.433820290985750},
    #     "gastroc":   {"Kp": 11.697677854044741, "Kd": 2.871243381043314},
    #     "soleus":    {"Kp": 9.698394821880356,  "Kd": 0.6579245587112381},
    #     "tib_ant":   {"Kp": 12.730049150690320, "Kd": 3.9676162677205866},
    # }

    

    # if USE_SYMMETRIC_PARAMS:
    #     base_muscles = MUSCLES[: len(MUSCLES) // 2]

    #     kp_array = np.array([
    #         SCONE_KP_KD[get_base_name(m)]["Kp"]
    #         for m in base_muscles
    #     ])

    #     kd_array = np.array([
    #         SCONE_KP_KD[get_base_name(m)]["Kd"]
    #         for m in base_muscles
    #     ])

    #     target_len_array = init_len[: len(MUSCLES) // 2].copy()

    # else:
    #     kp_array = np.array([
    #         SCONE_KP_KD[get_base_name(m)]["Kp"]
    #         for m in MUSCLES
    #     ])

    #     kd_array = np.array([
    #         SCONE_KP_KD[get_base_name(m)]["Kd"]
    #         for m in MUSCLES
    #     ])

    #     target_len_array = init_len.copy()

    # x0 = np.concatenate([
    #     kp_array,
    #     kd_array,
    #     target_len_array,
    # ])

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
            "controller": "StandingController + PDController",
            "use_symmetric_params": USE_SYMMETRIC_PARAMS,
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