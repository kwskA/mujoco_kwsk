# simulation/objectives/com_path_length_objective.py

import numpy as np

from simulation.objectives.base_objective import BaseObjective


class COMPathLengthObjective(BaseObjective):
    """
    COMの総軌跡長を評価するObjective。

    各stepのCOM位置の差分距離を累積し，
    その総移動距離をコストとして返す。

    gait1018では確認結果より，
        X = 前後方向
        Y = 左右方向
        Z = 鉛直方向
    とみなす。
    """

    def __init__(
        self,
        weight=1.0,
        name="com_path_length",
        axes=(0, 1, 2),
        apply_to="all",
    ):
        self.weight = float(weight)
        self.name = name
        self.axes = tuple(axes)
        self.apply_to = apply_to

        self.previous_com = None
        self.total_path_length = 0.0
        self.num_updates = 0

    def reset(self):
        self.previous_com = None
        self.total_path_length = 0.0
        self.num_updates = 0

    def update(
        self,
        step,
        time,
        model,
        data,
        ctrl,
        phase_name=None,
    ):
        current_com = data.subtree_com[0].copy()
        current_com = current_com[list(self.axes)]

        if self.previous_com is not None:
            diff = current_com - self.previous_com
            distance = float(np.linalg.norm(diff))
            self.total_path_length += distance

        self.previous_com = current_com
        self.num_updates += 1

    def finalize(self):
        return self.weight * self.total_path_length

    def get_log(self):
        return {
            "total_path_length": float(self.total_path_length),
            "weight": float(self.weight),
            "weighted_cost": float(self.weight * self.total_path_length),
            "axes": list(self.axes),
            "num_updates": int(self.num_updates),
        }