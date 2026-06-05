# control/pd_controller.py

import numpy as np

from control.base_controller import BaseController


class PDController(BaseController):
    """
    筋長に対するPD制御を行う制御器。
    """

    def __init__(self, model, muscles):

        self.model = model
        self.muscles = muscles

        self.num_muscles = len(muscles)

        self.actuator_ids = np.array([
            self.model.actuator(m).id
            for m in self.muscles
        ], dtype=int)

        self.tendon_ids = np.array([
            self.model.tendon(f"{m}_tendon").id
            for m in self.muscles
        ], dtype=int)

        self.Kp = None
        self.Kd = None
        self.target_length = None

    def set_params_from_vector(self, x):
        """
        最適化ベクトルをKp, Kd, target_lengthへ変換する。
        """

        x = np.asarray(x, dtype=float)

        n = self.num_muscles

        self.Kp = x[:n]
        self.Kd = x[n:2*n]
        self.target_length = x[2*n:3*n]

    def compute_ctrl(self, data, state_r=None, state_l=None):
        """
        筋長と筋速度から制御入力を計算する。
        """

        ctrl = np.zeros(self.model.nu)

        current_length = np.array([
            data.ten_length[tendon_id]
            for tendon_id in self.tendon_ids
        ])

        current_velocity = np.array([
            data.ten_velocity[tendon_id]
            for tendon_id in self.tendon_ids
        ])

        u = (
            self.Kp * (
                current_length - self.target_length
            )
            + self.Kd * current_velocity
        )

        u = np.clip(u, 0.0, 1.0)

        ctrl[self.actuator_ids] = u

        return ctrl

    def get_expected_param_dim(self):
        """
        必要な最適化パラメータ数を返す。
        """

        return 3 * self.num_muscles

    def make_initial_params(
        self,
        init_Kp=10.0,
        init_Kd=2.0,
        init_target_length=None,
    ):
        """
        初期パラメータを生成する。
        """

        if init_target_length is None:
            raise ValueError(
                "init_target_length is required"
            )

        return np.concatenate([
            np.full(self.num_muscles, init_Kp),
            np.full(self.num_muscles, init_Kd),
            np.asarray(init_target_length),
        ])