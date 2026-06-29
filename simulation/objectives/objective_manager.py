# simulation/objectives/objective_manager.py


class ObjectiveManager:
    """
    複数の評価関数をまとめて管理するクラス。
    """

    def __init__(
        self,
        objectives,
        total_cost_mode="sum",
    ):
        """
        使用する評価関数のリストと合計コストの計算方法を保存する。
        """

        self.objectives = objectives
        self.total_cost_mode = total_cost_mode

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

        details = {}

        for objective in self.objectives:
            details[objective.name] = objective.finalize()

        if self.total_cost_mode == "gait":
            simulation_time = details.get("simulation_time", 0.0)
            walking_speed = details.get("walking_speed", 0.0)

            total_cost = (
                max(simulation_time, walking_speed) * 100.0
                + walking_speed
            )

        elif self.total_cost_mode == "sum":
            total_cost = sum(details.values())

        else:
            raise ValueError(
                f"Unknown total_cost_mode: {self.total_cost_mode}"
            )

        return total_cost, details

    def get_logs(self):
        """
        すべての評価関数のログを辞書で返す。
        """

        logs = {}

        for objective in self.objectives:
            logs[objective.name] = objective.get_log()

        return logs