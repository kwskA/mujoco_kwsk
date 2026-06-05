# simulation/objectives/objective_manager.py


class ObjectiveManager:
    """
    複数の評価関数をまとめて管理するクラス。
    """

    def __init__(self, objectives):
        """
        使用する評価関数のリストを保存する。
        """
        self.objectives = objectives

    def reset(self):
        """
        すべての評価関数を初期化する。
        """
        for objective in self.objectives:
            objective.reset()

    def update(self, step, time, model, data, ctrl, phase_name=None):
        """
        すべての評価関数に現在stepの情報を渡す。
        """
        for objective in self.objectives:
            objective.update(
                step=step,
                time=time,
                model=model,
                data=data,
                ctrl=ctrl,
                phase_name=phase_name,
            )

    def finalize(self):
        """
        すべての評価関数を集計し、合計コストと個別コストを返す。
        """
        total_cost = 0.0
        details = {}

        for objective in self.objectives:
            value = objective.finalize()
            total_cost += value
            details[objective.name] = value

        return total_cost, details

    def get_logs(self):
        """
        すべての評価関数のログを辞書で返す。
        """
        logs = {}

        for objective in self.objectives:
            logs[objective.name] = objective.get_log()

        return logs