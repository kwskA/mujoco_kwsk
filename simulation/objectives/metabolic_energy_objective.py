# simulation/objectives/metabolic_energy_objective.py

import numpy as np

from simulation.objectives.base_objective import BaseObjective
from simulation.objectives.wang2012 import Wang2012


class MetabolicEnergyObjective(BaseObjective):
    """
    Wang2012に基づく代謝エネルギー評価関数。
    筋数が18でも54でも，muscle_namesに対応する筋データがあれば動作する。
    """

    def __init__(
        self,
        muscle_names,
        muscle_data_module,
        weight=1.0,
        name="metabolic_energy",
    ):
        self.name = name
        self.muscle_names = list(muscle_names)
        self.muscle_data_module = muscle_data_module
        self.weight = float(weight)

        self.mass = muscle_data_module.get_parameter_array(
            self.muscle_names,
            "muscle_mass",
        )

        self.l_mtu_opt = muscle_data_module.get_parameter_array(
            self.muscle_names,
            "l_mtu_opt",
        )

        self.slow_twitch_ratio = muscle_data_module.get_parameter_array(
            self.muscle_names,
            "slow_twitch_ratio",
        )

        self.reset()

    def reset(self):
        self.total_power_sum = 0.0
        self.total_energy_joule = 0.0
        self.step_count = 0

        self.last_total_power = 0.0
        self.mean_power = 0.0

    def update(self, step, time, model, data, ctrl, phase_name=None):
        total_power, muscle_power = Wang2012(
            u=ctrl,
            a=data.act,
            l_MTU=data.actuator_length,
            v_MTU=data.actuator_velocity,
            F_MTU=data.actuator_force,
            mass=self.mass,
            l_mtu_opt=self.l_mtu_opt,
            slow_twitch_ratio=self.slow_twitch_ratio,
        )

        dt = model.opt.timestep

        self.last_total_power = float(total_power)
        self.total_power_sum += float(total_power)
        self.total_energy_joule += float(total_power) * dt
        self.step_count += 1

        self.mean_power = self.total_power_sum / self.step_count

    def finalize(self):
        if self.step_count == 0:
            return 0.0

        return self.weight * self.mean_power

    def get_log(self):
        return {
            self.name: self.weight * self.mean_power,
            "metabolic_power": self.last_total_power,
            "mean_metabolic_power": self.mean_power,
            "metabolic_energy_joule": self.total_energy_joule,
        }