# simulation/objectives/base_objective.py


class BaseObjective:
    """
    評価関数の基底クラス。
    すべての評価関数で共通する関数を定義する。
    """

    def reset(self):
        """
        評価開始前に内部変数を初期化する。
        """
        raise NotImplementedError

    def update(self, step, time, model, data, ctrl, phase_name=None):
        """
        各シミュレーションstepで評価に必要な情報を更新する。
        """
        raise NotImplementedError

    def finalize(self):
        """
        シミュレーション終了後に最終的な評価値を返す。
        """
        raise NotImplementedError

    def get_log(self):
        """
        評価値のログを返す。
        """
        raise NotImplementedError
