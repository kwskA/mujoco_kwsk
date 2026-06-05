# optim/base_optimizer.py


class BaseOptimizer:
    """
    最適化手法の基底クラス。

    CMA-ES，PSO，GAなど，
    すべての最適化手法で共通する関数を定義する。
    """

    def optimize(self):
        """
        最適化を実行する。
        子クラスで具体的な処理を実装する。
        """
        raise NotImplementedError