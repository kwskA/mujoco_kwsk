# test_pool.py

import numpy as np
from pathlib import Path
from mujoco import MjModel, MjData, mj_forward

from control.control_methods.pd_controller import PDController
from control.phases.standing_controller import StandingController
from control.phases.gait_initiation_controller import GaitInitiationController
from control.phases.sequential_controller import SequentialController

from simulation.objectives.objective_manager import ObjectiveManager
from simulation.objectives.time_objective import SimulationTimeObjective
from simulation.objectives.walking_speed_objective import WalkingSpeedObjective

from simulation.pool import create_simulation_pool
from simulation.worker import run_simulation_worker


ROOT_DIR = Path(__file__).parent
MODEL_PATH = str(ROOT_DIR / "model" / "1018model.xml")

MUSCLES = [
    "hamstrings_r", "bifemsh_r", "glut_max_r",
    "iliopsoas_r", "rect_fem_r", "vasti_r",
    "gastroc_r", "soleus_r", "tib_ant_r",

    "hamstrings_l", "bifemsh_l", "glut_max_l",
    "iliopsoas_l", "rect_fem_l", "vasti_l",
    "gastroc_l", "soleus_l", "tib_ant_l",
]


def build_controller(model):
    """
    worker内で制御器を作成する。
    Windowsのmultiprocessingで使うため、必ずトップレベル関数にする。
    """

    standing = StandingController(
        name="Standing",
        control_method=PDController(model, MUSCLES),
    )

    gait_init = GaitInitiationController(
        name="GaitInitiation1",
        control_method=PDController(model, MUSCLES),
    )

    return SequentialController(
        phases=[standing, gait_init],
        durations=[0.5, None],
    )


def build_objective_manager():
    """
    worker内で評価関数群を作成する。
    """

    return ObjectiveManager([
        SimulationTimeObjective(
            target_steps=100,
            weight=20.0,
            apply_to="all",
        ),
        WalkingSpeedObjective(
            min_velocity=1.0,
            weight=1.0,
            apply_to=["GaitInitiation1"],
            qpos_index=0,
        ),
    ])


def main():
    model = MjModel.from_xml_path(MODEL_PATH)
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

    print("param_dim =", controller.get_expected_param_dim())
    print("x0 shape =", x0.shape)

    # テスト用に4個体だけ作る
    rng = np.random.default_rng(0)
    solutions = [
        x0 + rng.normal(0.0, 0.01, size=x0.shape)
        for _ in range(4)
    ]

    pool = create_simulation_pool(
        model_path=MODEL_PATH,
        sim_steps=100,
        controller_builder=build_controller,
        objective_manager_builder=build_objective_manager,
        initial_qpos=initial_qpos,
        n_jobs=2,
        reserve_cores=1,
    )

    try:
        results = pool.map(run_simulation_worker, solutions)

    finally:
        pool.close()
        pool.join()

    print("\n[Pool results]")

    for i, result in enumerate(results):
        print(f"\n--- solution {i + 1} ---")
        print("total_cost =", result["total_cost"])
        print("details =", result["details"])

    print("\n✅ worker.py / pool.py の並列実行テスト完了")


if __name__ == "__main__":
    main()