# test_plot_output.py

import numpy as np
from pathlib import Path

from mujoco import MjModel, MjData, mj_forward

from control.control_methods.pd_controller import PDController
from control.phases.standing_controller import StandingController
from control.phases.gait_initiation_controller import GaitInitiationController
from control.phases.sequential_controller import SequentialController

from simulation.runner import SimulationRunner

from simulation.objectives.objective_manager import ObjectiveManager
from simulation.objectives.time_objective import SimulationTimeObjective
from simulation.objectives.walking_speed_objective import WalkingSpeedObjective

from render.plot_objectives import ObjectivePlotter
from render.plot_muscles import MusclePlotter


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

    actuator_ids = np.array([
        model.actuator(m).id
        for m in MUSCLES
    ], dtype=int)

    init_len = np.array([
        data.ten_length[t_id]
        for t_id in tendon_ids
    ])

    qpos_names = [
        model.joint(i).name
        for i in range(model.njnt)
    ]

    qvel_names = [
        f"dof_{i}"
        for i in range(model.nv)
    ]

    standing = StandingController(
        name="Standing",
        control_method=PDController(model, MUSCLES),
    )

    gait_init = GaitInitiationController(
        name="GaitInitiation1",
        control_method=PDController(model, MUSCLES),
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

    runner = SimulationRunner(
        model=model,
        controller=controller,
        objective_manager=objective_manager,
        sim_steps=100,
        initial_qpos=initial_qpos,
        qpos_names=qpos_names,
        qvel_names=qvel_names,
        muscle_names=MUSCLES,
        tendon_ids=tendon_ids,
        actuator_ids=actuator_ids,
    )

    result = runner.run(data)

    sim_log = result["simulation_log"]

    save_dir = ROOT_DIR / "test_results" / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    objective_plotter = ObjectivePlotter(
        save_dir=str(save_dir / "objectives")
    )

    muscle_plotter = MusclePlotter(
        save_dir=str(save_dir / "muscles")
    )

    objective_plotter.plot(sim_log)
    muscle_plotter.plot_all(sim_log)

    print()
    print("[Plot output test]")
    print("log length =", sim_log.get_length())
    print("save_dir =", save_dir)

    print()
    print("created directories:")
    print(save_dir / "objectives")
    print(save_dir / "muscles")

    print()
    print("✅ plot出力テスト完了")


if __name__ == "__main__":
    main()