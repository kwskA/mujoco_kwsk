# simulation/worker.py

import numpy as np
from mujoco import MjModel, MjData

from simulation.runner import SimulationRunner


_WORKER = {}


def _init_worker(
    model_path,
    sim_steps,
    controller_builder,
    objective_manager_builder,
    initial_qpos=None,
    fall_detector_builder=None,
):
    """
    並列計算用workerを初期化する。
    各プロセスで一度だけモデル・制御器・評価関数群・転倒判定器を作成する。
    """

    model = MjModel.from_xml_path(model_path)

    controller = controller_builder(model)
    objective_manager = objective_manager_builder()

    fall_detector = None

    if fall_detector_builder is not None:
        fall_detector = fall_detector_builder(model)

    _WORKER["model"] = model
    _WORKER["sim_steps"] = sim_steps
    _WORKER["controller"] = controller
    _WORKER["objective_manager"] = objective_manager
    _WORKER["initial_qpos"] = initial_qpos
    _WORKER["fall_detector"] = fall_detector


def run_simulation_worker(params):
    """
    最適化アルゴリズムから渡された1個体のパラメータを評価する。
    """

    model = _WORKER["model"]
    sim_steps = _WORKER["sim_steps"]
    controller = _WORKER["controller"]
    objective_manager = _WORKER["objective_manager"]
    initial_qpos = _WORKER["initial_qpos"]
    fall_detector = _WORKER["fall_detector"]

    params = np.asarray(params, dtype=float)

    controller.set_params_from_vector(params)

    data = MjData(model)

    runner = SimulationRunner(
        model=model,
        controller=controller,
        objective_manager=objective_manager,
        sim_steps=sim_steps,
        initial_qpos=initial_qpos,
        fall_detector=fall_detector,
        enable_log=False,
    )

    result = runner.run(data)

    total_cost, details = objective_manager.finalize()

    return {
        "total_cost": float(total_cost),
        "details": details,
        "survived_steps": result["survived_steps"],
        "fallen": result["fallen"],
    }