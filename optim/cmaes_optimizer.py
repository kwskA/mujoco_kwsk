# optim/cmaes_optimizer.py

import os
import numpy as np
from cma import CMAEvolutionStrategy

from optim.base_optimizer import BaseOptimizer
from optim.checkpoint_manager import CheckpointManager
from optim.optimization_logger import OptimizationLogger

from simulation.pool import create_simulation_pool
from simulation.worker import run_simulation_worker


class CMAESOptimizer(BaseOptimizer):
    """
    CMA-ESによる最適化を行うクラス。
    """

    def __init__(
        self,
        model_path,
        controller,
        controller_builder,
        objective_manager_builder,
        sim_steps,
        result_dir,
        initial_qpos=None,
        muscle_names=None,
        sigma0=0.5,
        popsize=25,
        maxiter=200,
        n_jobs=None,
        reserve_cores=1,
        bounds_lower=None,
        bounds_upper=None,
        checkpoint_interval=500,
        write_checkpoint_video=True,
        write_checkpoint_plots=True,
        write_checkpoint_csv=True,
        fall_detector_builder=None,
    ):
        """
        CMA-ESに必要な設定と保存機構を初期化する。
        """

        self.model_path = model_path

        self.controller = controller
        self.controller_builder = controller_builder
        self.objective_manager_builder = objective_manager_builder

        self.sim_steps = int(sim_steps)
        self.result_dir = result_dir
        self.initial_qpos = initial_qpos
        self.muscle_names = muscle_names

        self.sigma0 = float(sigma0)
        self.popsize = int(popsize)
        self.maxiter = int(maxiter)

        self.n_jobs = n_jobs
        self.reserve_cores = int(reserve_cores)

        self.bounds_lower = bounds_lower
        self.bounds_upper = bounds_upper

        self.fall_detector_builder = fall_detector_builder

        self.checkpoint_interval = checkpoint_interval
        self.write_checkpoint_video = write_checkpoint_video
        self.write_checkpoint_plots = write_checkpoint_plots
        self.write_checkpoint_csv = write_checkpoint_csv

        self.cost_history_min = []
        self.cost_history_mean = []
        self.best_params = None
        self.best_cost = np.inf
        self.best_details = {}

        os.makedirs(self.result_dir, exist_ok=True)

        self.optimization_logger = OptimizationLogger(
            result_dir=self.result_dir,
        )

        self.checkpoint_manager = CheckpointManager(
            result_dir=self.result_dir,
            model_path=self.model_path,
            controller_builder=self.controller_builder,
            objective_manager_builder=self.objective_manager_builder,
            sim_steps=self.sim_steps,
            initial_qpos=self.initial_qpos,
            muscle_names=self.muscle_names,
        )

    def optimize(self, x0):
        """
        CMA-ESを実行し，最良パラメータとログを返す。
        """

        x0 = np.asarray(x0, dtype=float)

        opts = {
            "maxiter": self.maxiter,
            "popsize": self.popsize,
            "tolfun": 1e-12,
            "tolx": 1e-12,
            "tolfunhist": 1e-12,

            # "seed": 903252,

        }

        bounds_lower, bounds_upper = self._prepare_bounds(x0)

        opts["bounds"] = [
            bounds_lower,
            bounds_upper,
        ]

        es = CMAEvolutionStrategy(
            x0,
            self.sigma0,
            opts,
        )

        pool = create_simulation_pool(
            model_path=self.model_path,
            sim_steps=self.sim_steps,
            controller_builder=self.controller_builder,
            objective_manager_builder=self.objective_manager_builder,
            initial_qpos=self.initial_qpos,
            fall_detector_builder=self.fall_detector_builder,
            n_jobs=self.n_jobs,
            reserve_cores=self.reserve_cores,
        )

        try:
            while not es.stop():

                solutions = es.ask()

                results = pool.map(
                    run_simulation_worker,
                    solutions,
                )

                costs = np.array([
                    result["total_cost"]
                    for result in results
                ], dtype=float)

                es.tell(
                    solutions,
                    costs,
                )

                es.disp()

                generation = int(es.countiter)

                min_idx = int(np.argmin(costs))
                generation_min_cost = float(np.min(costs))
                generation_mean_cost = float(np.mean(costs))

                generation_best_cost = float(costs[min_idx])
                generation_best_params = np.asarray(
                    solutions[min_idx],
                    dtype=float,
                )

                generation_best_details = results[min_idx].get(
                    "details",
                    {},
                )

                if generation_best_cost < self.best_cost:
                    self.best_cost = generation_best_cost
                    self.best_params = generation_best_params.copy()
                    self.best_details = dict(generation_best_details)

                self.cost_history_min.append(generation_min_cost)
                self.cost_history_mean.append(generation_mean_cost)

                self.optimization_logger.record(
                    generation=generation,
                    min_cost=generation_min_cost,
                    mean_cost=generation_mean_cost,
                    best_cost=self.best_cost,
                    details=self.best_details,
                )

                # print(
                #     f"[CMAESOptimizer] generation={generation}, "
                #     f"generation_min={generation_min_cost}, "
                #     f"mean={generation_mean_cost}, "
                #     f"global_best={self.best_cost}"
                # )

                if self._should_save_checkpoint(generation):
                    self._save_checkpoint(
                        generation=generation,
                        tag=f"gen_{generation:04d}",
                    )

        except KeyboardInterrupt:
            print()
            print("[CMAESOptimizer] KeyboardInterrupt detected.")
            print("[CMAESOptimizer] saving current best before exit.")

            if self.best_params is not None:
                self._save_checkpoint(
                    generation=int(es.countiter),
                    tag="interrupted",
                )

        finally:
            pool.close()
            pool.join()

            self.optimization_logger.save_all()

        if self.best_params is None:
            self.best_params = np.asarray(
                es.result.xbest,
                dtype=float,
            )
            self.best_cost = float(es.result.fbest)

        self._save_checkpoint(
            generation=int(es.countiter),
            tag="final",
        )

        return {
            "best_params": self.best_params,
            "best_cost": self.best_cost,
            "best_details": self.best_details,
            "cost_history_min": self.cost_history_min,
            "cost_history_mean": self.cost_history_mean,
        }

    def _should_save_checkpoint(self, generation):
        """
        現在世代でcheckpointを保存するか判定する。
        """

        if self.checkpoint_interval is None:
            return False

        if self.checkpoint_interval <= 0:
            return False

        if self.best_params is None:
            return False

        return generation % self.checkpoint_interval == 0

    def _make_default_bounds(self, x0):
        param_dim = x0.size

        if param_dim % 3 != 0:
            raise ValueError(
                f"Parameter dimension must be divisible by 3, got {param_dim}"
            )

        n = param_dim // 3

        bounds_lower = np.concatenate([
            np.full(n, 0.0),     # Kp
            np.full(n, 0.0),     # Kd
            np.full(n, 0.00),    # target_length
        ])

        bounds_upper = np.concatenate([
            np.full(n, 10.0),    # Kp
            np.full(n, 2.0),     # Kd
            np.full(n, 2.0),     # target_length
        ])

        return bounds_lower, bounds_upper

    def _prepare_bounds(self, x0):
        """
        x0の次元に合わせてCMA-ESのboundsを準備する。
        明示的なboundsが与えられていない場合は自動生成する。
        """

        if self.bounds_lower is None or self.bounds_upper is None:
            bounds_lower, bounds_upper = self._make_default_bounds(x0)
        else:
            bounds_lower = np.asarray(self.bounds_lower, dtype=float)
            bounds_upper = np.asarray(self.bounds_upper, dtype=float)

            if bounds_lower.size != x0.size or bounds_upper.size != x0.size:
                print("[CMAESOptimizer] bounds size mismatch.")
                print(f"  x0.size = {x0.size}")
                print(f"  bounds_lower.size = {bounds_lower.size}")
                print(f"  bounds_upper.size = {bounds_upper.size}")
                print("[CMAESOptimizer] Using auto-generated bounds instead.")

                bounds_lower, bounds_upper = self._make_default_bounds(x0)

        if np.any(x0 < bounds_lower) or np.any(x0 > bounds_upper):
            bad_indices = np.where(
                (x0 < bounds_lower) | (x0 > bounds_upper)
            )[0]

            raise ValueError(
                "Initial solution x0 is outside bounds.\n"
                f"Bad indices: {bad_indices.tolist()}\n"
                f"x0[bad]: {x0[bad_indices]}\n"
                f"lower[bad]: {bounds_lower[bad_indices]}\n"
                f"upper[bad]: {bounds_upper[bad_indices]}"
            )

        return bounds_lower, bounds_upper

    def _save_checkpoint(
        self,
        generation,
        tag,
    ):
        """
        現在の最良解をcheckpointとして保存する。
        """

        if self.best_params is None:
            print("[CMAESOptimizer] no best_params, checkpoint skipped.")
            return

        self.checkpoint_manager.save_checkpoint(
            generation=generation,
            best_cost=self.best_cost,
            best_params=self.best_params,
            tag=tag,
            write_video=self.write_checkpoint_video,
            write_plots=self.write_checkpoint_plots,
            write_csv=self.write_checkpoint_csv,
        )
