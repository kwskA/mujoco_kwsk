# simulation/objectives/walking_speed_objective.py

from simulation.objectives.base_objective import BaseObjective


class WalkingSpeedObjective(BaseObjective):
    """
    歩行速度が最低目標速度を下回る場合にペナルティを与える評価関数。
    """

    def __init__(
        self,
        min_velocity=1.0,
        weight=1.0,
        apply_to=None,
        qpos_index=0,
    ):
        """
        最低速度、重み、適用フェーズ、前進方向qpos番号を設定する。
        """
        self.name = "walking_speed"

        self.min_velocity = float(min_velocity)
        self.weight = float(weight)
        self.apply_to = apply_to
        self.qpos_index = int(qpos_index)

        self.reset()

    def reset(self):
        """
        評価対象フェーズの開始位置・終了位置・時間を初期化する。
        """
        self.started = False

        self.start_pos = None
        self.end_pos = None

        self.start_time = None
        self.end_time = None

        self.speed_log = []
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
        評価対象フェーズにおける開始位置・終了位置・時刻を記録する。
        """
        if not self._is_applicable(phase_name):
            return

        current_pos = float(data.qpos[self.qpos_index])
        current_time = float(time)

        if not self.started:
            self.started = True
            self.start_pos = current_pos
            self.start_time = current_time

        self.end_pos = current_pos
        self.end_time = current_time

        speed = self._compute_speed()
        cost = self._compute_cost(speed)

        self.speed_log.append(speed)
        self.cost_log.append(cost)

    def _compute_speed(self):
        """
        現在までの平均歩行速度を計算する。
        """
        if not self.started:
            return 0.0

        duration = max(
            self.end_time - self.start_time,
            1e-6,
        )

        return (self.end_pos - self.start_pos) / duration

    def _compute_cost(self, speed):
        """
        最低速度を下回った分のペナルティを計算する。
        """
        if self.min_velocity <= 0.0:
            return 0.0

        penalty = max(
            0.0,
            (self.min_velocity - speed) / self.min_velocity,
        )

        return self.weight * penalty

    def finalize(self):
        """
        最終的な歩行速度ペナルティを返す。
        """
        speed = self._compute_speed()
        return self._compute_cost(speed)

    def get_log(self):
        """
        歩行速度評価のログを返す。
        """
        final_speed = self._compute_speed()
        final_cost = self._compute_cost(final_speed)

        return {
            "name": self.name,
            "min_velocity": self.min_velocity,
            "weight": self.weight,
            "apply_to": self.apply_to,
            "qpos_index": self.qpos_index,
            "final_speed": final_speed,
            "final_cost": final_cost,
            "speed_log": self.speed_log,
            "cost_log": self.cost_log,
        }