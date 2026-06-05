# simulation/objectives/simulation_time_objective.py

from simulation.objectives.base_objective import BaseObjective


class SimulationTimeObjective(BaseObjective):
    """
    シミュレーション継続時間を評価するクラス。

    目的:
        長くシミュレーションできた場合は低コスト、
        早く終了・転倒した場合は高コストにする。
    """

    def __init__(
        self,
        target_steps,
        weight=1.0,
        apply_to=None,
    ):
        """
        目標step数、重み、適用する動作フェーズを設定する。
        """
        self.name = "simulation_time"
        self.target_steps = int(target_steps)
        self.weight = float(weight)
        self.apply_to = apply_to

        self.reset()

    def reset(self):
        """
        評価用のstep数とログを初期化する。
        """

        self.evaluated_steps = 0
        self.last_step = 0
        self.cost_log = []

    def _is_applicable(self, phase_name):
        """
        現在の動作フェーズが評価対象か判定する。
        """

        if self.apply_to is None:
            return True

        if self.apply_to == "all":
            return True

        return phase_name in self.apply_to

    def update(self, step, time, model, data, ctrl, phase_name=None):
        """
        評価対象フェーズのシミュレーションstep数を数える。
        """

        self.last_step = step

        if self._is_applicable(phase_name):
            self.evaluated_steps += 1

        current_cost = self._compute_cost()
        self.cost_log.append(current_cost)

    def _compute_cost(self):
        """
        現時点でのシミュレーション時間コストを計算する。
        """

        if self.target_steps <= 0:
            return 0.0

        survival_ratio = self.evaluated_steps / self.target_steps
        survival_ratio = min(max(survival_ratio, 0.0), 1.0)

        cost = 1.0 - survival_ratio

        return self.weight * cost

    def finalize(self):
        """
        最終的なシミュレーション時間コストを返す。
        """

        return self._compute_cost()

    def get_log(self):
        """
        シミュレーション時間評価のログを返す。
        """

        return {
            "name": "simulation_time",
            "evaluated_steps": self.evaluated_steps,
            "target_steps": self.target_steps,
            "cost_log": self.cost_log,
            "final_cost": self.finalize(),
        }
