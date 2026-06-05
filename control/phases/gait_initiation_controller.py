# control/phases/gait_initiation_controller.py

from control.base_controller import BaseController


class GaitInitiationController(BaseController):
    """
    歩行開始動作フェーズを担当する制御器。
    同じクラスからGaitInitiation1, 2, 3を複製して使う。
    """

    def __init__(self, name, control_method):
        """
        フェーズ名と使用する制御方法を保存する。
        """
        self.name = name
        self.control_method = control_method

    def set_params_from_vector(self, x):
        """
        最適化ベクトルを内部の制御方法に渡す。
        """
        self.control_method.set_params_from_vector(x)

    def compute_ctrl(self, data, state_r=None, state_l=None):
        """
        歩行開始中のMuJoCo制御入力を計算する。
        """
        return self.control_method.compute_ctrl(data)

    def get_expected_param_dim(self):
        """
        歩行開始制御に必要な最適化パラメータ数を返す。
        """
        return self.control_method.get_expected_param_dim()

    def make_initial_params(self, *args, **kwargs):
        """
        歩行開始制御用の初期パラメータを作成する。
        """
        return self.control_method.make_initial_params(*args, **kwargs)