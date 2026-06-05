# simulation/logs/simulation_log.py

import numpy as np


class SimulationLog:
    """
    1回のシミュレーション中に得られる時系列データを保存するクラス。
    """

    def __init__(
        self,
        qpos_names=None,
        qvel_names=None,
        muscle_names=None,
        objective_names=None,
    ):
        self.qpos_names = list(qpos_names) if qpos_names is not None else []
        self.qvel_names = list(qvel_names) if qvel_names is not None else []
        self.muscle_names = list(muscle_names) if muscle_names is not None else []
        self.objective_names = list(objective_names) if objective_names is not None else []

        self.reset()

    def reset(self):
        self.time = []
        self.step = []
        self.phase_name = []

        self.qpos = []
        self.qvel = []

        self.com = []
        self.cop = []
        self.grf = []

        self.tendon_length = []
        self.tendon_velocity = []
        self.actuator_force = []
        self.activation = []

        self.target_length = []

        self.objective_values = []

    def record_step(
        self,
        step,
        time,
        phase_name,
        qpos,
        qvel,
        com=None,
        cop=None,
        grf=None,
        tendon_length=None,
        tendon_velocity=None,
        actuator_force=None,
        activation=None,
        target_length=None,
        objective_values=None,
    ):
        self.step.append(int(step))
        self.time.append(float(time))
        self.phase_name.append(phase_name)

        self.qpos.append(np.asarray(qpos, dtype=float).copy())
        self.qvel.append(np.asarray(qvel, dtype=float).copy())

        self.com.append(self._to_array_or_nan(com, 3))
        self.cop.append(self._to_array_or_nan(cop, 3))
        self.grf.append(self._to_array_or_nan(grf, 3))

        self.tendon_length.append(self._to_array_or_empty(tendon_length))
        self.tendon_velocity.append(self._to_array_or_empty(tendon_velocity))
        self.actuator_force.append(self._to_array_or_empty(actuator_force))
        self.activation.append(self._to_array_or_empty(activation))

        self.target_length.append(self._to_array_or_empty(target_length))

        if objective_values is None:
            objective_values = {}

        self.objective_values.append(dict(objective_values))

    def _to_array_or_nan(self, value, size):
        if value is None:
            return np.full(size, np.nan, dtype=float)

        return np.asarray(value, dtype=float).copy()

    def _to_array_or_empty(self, value):
        if value is None:
            return np.asarray([], dtype=float)

        return np.asarray(value, dtype=float).copy()

    def get_length(self):
        return len(self.time)

    def to_dict(self):
        return {
            "qpos_names": self.qpos_names,
            "qvel_names": self.qvel_names,
            "muscle_names": self.muscle_names,
            "objective_names": self.objective_names,

            "step": self.step,
            "time": self.time,
            "phase_name": self.phase_name,

            "qpos": self.qpos,
            "qvel": self.qvel,

            "com": self.com,
            "cop": self.cop,
            "grf": self.grf,

            "tendon_length": self.tendon_length,
            "tendon_velocity": self.tendon_velocity,
            "actuator_force": self.actuator_force,
            "activation": self.activation,
            "target_length": self.target_length,

            "objective_values": self.objective_values,
        }