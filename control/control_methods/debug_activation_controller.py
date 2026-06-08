import numpy as np

from control.base_controller import BaseController


class DebugActivationController(BaseController):
    """
    筋ごとのactivationを直接指定するデバッグ用コントローラ。
    """

    def __init__(
        self,
        model,
        muscles,
        activation_dict=None,
        default_activation=1.0,
    ):
        self.model = model
        self.muscles = list(muscles)

        self.activation_dict = activation_dict or {}
        self.default_activation = float(default_activation)

        self.actuator_ids = np.array([
            self.model.actuator(m).id
            for m in self.muscles
        ], dtype=int)

        self.activation_values = self._build_activation_values()

    def _build_activation_values(self):
        values = []

        for muscle in self.muscles:
            value = self.activation_dict.get(
                muscle,
                self.default_activation,
            )

            value = float(value)
            value = np.clip(value, 0.0, 1.0)

            values.append(value)

        return np.asarray(values, dtype=float)

    def compute_ctrl(self, data, state_r=None, state_l=None):
        ctrl = np.zeros(self.model.nu)

        ctrl[self.actuator_ids] = self.activation_values

        return ctrl