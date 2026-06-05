# simulation/pool.py

import os
from multiprocessing import Pool

from simulation.worker import _init_worker


def resolve_n_jobs(
    n_jobs=None,
    reserve_cores=1,
):
    """
    実際に使用する並列worker数を決定する。

    reserve_cores:
        動画出力やファイル書き出し用に空けておくCPUコア数。
    """

    cpu_count = os.cpu_count()

    if cpu_count is None:
        cpu_count = 1

    max_jobs = max(
        1,
        cpu_count - int(reserve_cores),
    )

    if n_jobs is None:
        return max_jobs

    return max(
        1,
        min(int(n_jobs), max_jobs),
    )


def create_simulation_pool(
    model_path,
    sim_steps,
    controller_builder,
    objective_manager_builder,
    initial_qpos=None,
    n_jobs=None,
    reserve_cores=1,
    fall_detector_builder=None,
):
    """
    シミュレーション評価用のmultiprocessing.Poolを作成する。

    各workerでは，
    モデル・制御器・評価関数群を一度だけ初期化する。
    """

    resolved_jobs = resolve_n_jobs(
        n_jobs=n_jobs,
        reserve_cores=reserve_cores,
    )

    print(
        f"[simulation_pool] cpu_count={os.cpu_count()}, "
        f"reserve_cores={reserve_cores}, "
        f"n_jobs={resolved_jobs}"
    )

    pool = Pool(
        processes=resolved_jobs,
        initializer=_init_worker,
        initargs=(
            model_path,
            sim_steps,
            controller_builder,
            objective_manager_builder,
            initial_qpos,
            fall_detector_builder,
        ),
    )

    return pool