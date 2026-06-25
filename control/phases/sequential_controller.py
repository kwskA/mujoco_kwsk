# control/phases/sequential_controller.py

import numpy as np

from control.base_controller import BaseController


class SequentialController(BaseController):
    """
    時間に応じて複数の動作フェーズを順番に切り替える制御器。
    例: Standing → GaitInitiation1 → GaitInitiation2 → Gait
    """

    def __init__(self, phases, durations):
        """
        フェーズ一覧と各フェーズの継続時間を保存する。
        durationsの最後はNoneでもよい。
        """
        self.phases = phases
        self.durations = durations

        if len(self.phases) != len(self.durations):
            raise ValueError("phases and durations must have the same length.")

    def reset(self):
        """
        各フェーズの内部状態を初期化する。
        """
        for phase in self.phases:
            if hasattr(phase, "reset"):
                phase.reset()

    def _get_phase_index(self, time):
        """
        現在時刻から使用するフェーズ番号を返す。
        """
        elapsed = 0.0

        for i, duration in enumerate(self.durations):
            if duration is None:
                return i

            elapsed += duration

            if time < elapsed:
                return i

        return len(self.phases) - 1

    def get_current_phase(self, time):
        """
        現在時刻で使用するフェーズを返す。
        """
        phase_index = self._get_phase_index(time)
        return self.phases[phase_index]

    def set_params_from_vector(self, x):
        """
        連結された最適化ベクトルを各フェーズに分配する。
        """
        x = np.asarray(x, dtype=float)

        offset = 0

        for phase in self.phases:
            dim = phase.get_expected_param_dim()

            phase_params = x[offset: offset + dim]
            phase.set_params_from_vector(phase_params)

            offset += dim

        if offset != x.size:
            raise ValueError(
                f"Parameter size mismatch: used {offset}, got {x.size}"
            )

    def compute_ctrl(self, data, state_r=None, state_l=None):
        """
        現在時刻に対応するフェーズの制御入力を計算する。
        """
        phase = self.get_current_phase(data.time)
        return phase.compute_ctrl(data, state_r=state_r, state_l=state_l)

    def get_expected_param_dim(self):
        """
        全フェーズを合わせた最適化パラメータ数を返す。
        """
        return sum(
            phase.get_expected_param_dim()
            for phase in self.phases
        )

    def make_initial_params(self, *args, **kwargs):
        """
        全フェーズの初期パラメータを連結して返す。
        """
        return np.concatenate([
            phase.make_initial_params(*args, **kwargs)
            for phase in self.phases
        ])
