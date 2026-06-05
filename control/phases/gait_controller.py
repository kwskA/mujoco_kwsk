# control/phases/gait_controller.py

from control.base_controller import BaseController
from control.phases.gait.gait_state_controller import GaitStateController


class GaitController(BaseController):
    """
    歩行動作フェーズを担当する制御器。
    歩行相を判定し、その歩行相に応じて制御入力を計算する。
    """

    def __init__(self, name, model, control_method):
        """
        フェーズ名、MuJoCoモデル、制御方法、歩行相判定器を保存する。
        """
        self.name = name
        self.model = model
        self.control_method = control_method
        self.gait_state_controller = GaitStateController(model)

    def set_params_from_vector(self, x):
        """
        最適化ベクトルを内部の制御方法に渡す。
        """
        self.control_method.set_params_from_vector(x)

    def compute_ctrl(self, data, state_r=None, state_l=None):
        """
        現在の歩行相を判定し、MuJoCo制御入力を計算する。
        """
        state_r, state_l = self.gait_state_controller.update(data)

        return self.control_method.compute_ctrl(
            data,
            state_r=state_r,
            state_l=state_l,
        )

    def get_expected_param_dim(self):
        """
        歩行制御に必要な最適化パラメータ数を返す。
        """
        return self.control_method.get_expected_param_dim()

    def make_initial_params(self, *args, **kwargs):
        """
        歩行制御用の初期パラメータを作成する。
        """
        return self.control_method.make_initial_params(*args, **kwargs)