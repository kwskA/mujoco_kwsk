# control/base_controller.py

class BaseController:

    # 最適化ベクトルxを制御器内部のパラメータに変換する
    def set_params_from_vector(self, x):
        raise NotImplementedError
    
    # 現在のシミュレーション状態から MuJoCo の制御入力 ctrl を計算する
    def compute_ctrl(self, data, state_r=None, state_l=None):
        raise NotImplementedError

    # 制御器が必要とする最適化パラメータ数を返す
    def get_expected_param_dim(self):
        raise NotImplementedError
    
    # CMA-ESなどの最適化で使う初期パラメータ x0 を作成する
    def make_initial_params(self):
        raise NotImplementedError