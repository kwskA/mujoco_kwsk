# render/plot_objectives.py

import os
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


class ObjectivePlotter:
    """
    評価関数の時系列ログをグラフとして保存するクラス。
    """

    def __init__(self, save_dir):
        """
        グラフ保存先ディレクトリを設定する。
        """

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot(self, sim_log, filename="objectives.png"):
        """
        SimulationLog内のobjective値を時系列グラフとして保存する。
        """

        if sim_log.get_length() == 0:
            print("[ObjectivePlotter] empty log, skipped.")
            return

        if len(sim_log.objective_names) == 0:
            print("[ObjectivePlotter] no objective names, skipped.")
            return

        save_path = os.path.join(self.save_dir, filename)

        plt.figure(figsize=(10, 5))

        for objective_name in sim_log.objective_names:
            values = []

            for objective_values in sim_log.objective_values:
                values.append(
                    objective_values.get(objective_name, float("nan"))
                )

            plt.plot(
                sim_log.time,
                values,
                label=objective_name,
            )

        plt.xlabel("Time [s]")
        plt.ylabel("Objective value")
        plt.title("Objective history")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(save_path)
        plt.close()

        # print(f"[ObjectivePlotter] wrote {save_path}")