# test_simulation.py

import numpy as np
from mujoco import MjModel, MjData, mj_forward

from control.control_methods.pd_controller import PDController
from control.phases.standing_controller import StandingController
from control.phases.gait_initiation_controller import GaitInitiationController
from control.phases.sequential_controller import SequentialController

from simulation.runner import SimulationRunner
from simulation.objectives.objective_manager import ObjectiveManager
from simulation.objectives.time_objective import SimulationTimeObjective
from simulation.objectives.walking_speed_objective import WalkingSpeedObjective


def main():
    model_path = "model/1018model.xml"

    muscles = [
        "hamstrings_r", "bifemsh_r", "glut_max_r",
        "iliopsoas_r", "rect_fem_r", "vasti_r",
        "gastroc_r", "soleus_r", "tib_ant_r",

        "hamstrings_l", "bifemsh_l", "glut_max_l",
        "iliopsoas_l", "rect_fem_l", "vasti_l",
        "gastroc_l", "soleus_l", "tib_ant_l",
    ]

    model = MjModel.from_xml_path(model_path)
    data = MjData(model)

    data.qpos[:] = model.key_qpos[0].copy()
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mj_forward(model, data)

    tendon_ids = np.array([
        model.tendon(f"{m}_tendon").id
        for m in muscles
    ], dtype=int)

    init_len = np.array([
        data.ten_length[t_id]
        for t_id in tendon_ids
    ])

    # -------------------------
    # controller
    # -------------------------

    standing = StandingController(
        name="Standing",
        control_method=PDController(model, muscles),
    )

    gait_init = GaitInitiationController(
        name="GaitInitiation1",
        control_method=PDController(model, muscles),
    )

    controller = SequentialController(
        phases=[standing, gait_init],
        durations=[0.5, None],
    )

    x0 = controller.make_initial_params(
        init_Kp=10.0,
        init_Kd=2.0,
        init_target_length=init_len,
    )

    controller.set_params_from_vector(x0)

    # -------------------------
    # objectives
    # -------------------------

    objective_manager = ObjectiveManager([
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

    # -------------------------
    # runner
    # -------------------------

    runner = SimulationRunner(
        model=model,
        controller=controller,
        objective_manager=objective_manager,
        sim_steps=100,
        initial_qpos=model.key_qpos[0].copy(),
    )

    result = runner.run(data)

    total_cost, details = objective_manager.finalize()
    logs = objective_manager.get_logs()

    print("\n[SimulationRunner]")
    print("survived_steps =", result["survived_steps"])
    print("sim_steps =", result["sim_steps"])
    print("fallen =", result["fallen"])

    print("\n[ObjectiveManager]")
    print("total_cost =", total_cost)
    print("details =", details)

    print("\n[Logs]")
    for name, log in logs.items():
        print(name, "final_cost =", log["final_cost"])

    print("\n✅ simulationフォルダの基本テスト完了")


if __name__ == "__main__":
    main()