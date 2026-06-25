# simulation/objectives/walking_speed_objective.py

from simulation.objectives.base_objective import BaseObjective


class WalkingSpeedObjective(BaseObjective):
    """
    平均歩行速度が理想速度範囲から外れた場合にペナルティを与える評価関数。
    """

    def __init__(
        self,
        min_velocity=0.2,
        ideal_velocity=1.0,
        max_velocity=1.8,
        weight=1.0,
        apply_to=None,
        qpos_index=0,
        ideal_margin=0.3,
    ):
        self.name = "walking_speed"

        self.min_velocity = float(min_velocity)
        self.ideal_velocity = float(ideal_velocity)
        self.max_velocity = float(max_velocity)
        self.ideal_margin = float(ideal_margin)

        self.weight = float(weight)
        self.apply_to = apply_to
        self.qpos_index = int(qpos_index)

        self._validate_velocity_settings()
        self.reset()

    def _validate_velocity_settings(self):
        lower_ideal = self.ideal_velocity - self.ideal_margin
        upper_ideal = self.ideal_velocity + self.ideal_margin

        if not self.min_velocity < lower_ideal:
            raise ValueError(
                "min_velocity must be smaller than "
                "ideal_velocity - ideal_margin."
            )

        if not upper_ideal < self.max_velocity:
            raise ValueError(
                "max_velocity must be larger than "
                "ideal_velocity + ideal_margin."
            )

    def reset(self):
        self.started = False

        self.start_pos = None
        self.end_pos = None

        self.start_time = None
        self.end_time = None

        self.speed_log = []
        self.cost_log = []

    def _is_applicable(self, phase_name):
        if self.apply_to is None:
            return True

        if self.apply_to == "all":
            return True

        return phase_name in self.apply_to

    def update(self, step, time, model, data, ctrl, phase_name=None):
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
        if not self.started:
            return 0.0

        duration = max(
            self.end_time - self.start_time,
            1e-6,
        )

        return (self.end_pos - self.start_pos) / duration

    def _compute_cost(self, speed):
        lower_ideal = self.ideal_velocity - self.ideal_margin
        upper_ideal = self.ideal_velocity + self.ideal_margin

        if speed <= self.min_velocity:
            penalty = 1.0

        elif speed <= lower_ideal:
            penalty = (
                (lower_ideal - speed)
                / (lower_ideal - self.min_velocity)
            )

        elif speed <= upper_ideal:
            penalty = 0.0

        elif speed < self.max_velocity:
            penalty = (
                (speed - upper_ideal)
                / (self.max_velocity - upper_ideal)
            )

        else:
            penalty = 1.0

        penalty = max(0.0, min(1.0, penalty))

        return self.weight * penalty

    def finalize(self):
        speed = self._compute_speed()
        return self._compute_cost(speed)

    def get_log(self):
        final_speed = self._compute_speed()
        final_cost = self._compute_cost(final_speed)

        return {
            "name": self.name,
            "min_velocity": self.min_velocity,
            "ideal_velocity": self.ideal_velocity,
            "max_velocity": self.max_velocity,
            "ideal_margin": self.ideal_margin,
            "weight": self.weight,
            "apply_to": self.apply_to,
            "qpos_index": self.qpos_index,
            "final_speed": final_speed,
            "final_cost": final_cost,
            "speed_log": self.speed_log,
            "cost_log": self.cost_log,
        }