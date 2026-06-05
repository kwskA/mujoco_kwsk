# render/plot_muscles.py

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


class MusclePlotter:
    """
    各筋について target_length, actual_length, activation をグラフ化するクラス。
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_all(self, sim_log):
        """
        全筋についてグラフを保存する。
        """

        for muscle_index, muscle_name in enumerate(sim_log.muscle_names):
            self.plot_one(
                sim_log=sim_log,
                muscle_index=muscle_index,
                muscle_name=muscle_name,
            )

    def plot_one(
        self,
        sim_log,
        muscle_index,
        muscle_name,
    ):
        """
        1つの筋について target_length, actual_length, activation を保存する。
        """

        if sim_log.get_length() == 0:
            print("[MusclePlotter] empty log, skipped.")
            return

        time = np.asarray(sim_log.time, dtype=float)

        actual_length = self._extract_series(
            sim_log.tendon_length,
            muscle_index,
        )

        target_length = self._extract_series(
            sim_log.target_length,
            muscle_index,
        )

        activation = self._extract_series(
            sim_log.activation,
            muscle_index,
        )

        save_path = os.path.join(
            self.save_dir,
            f"{muscle_name}.png",
        )

        plt.figure(figsize=(10, 5))

        plt.plot(time, actual_length, label="actual_length")
        plt.plot(time, target_length, label="target_length")
        plt.plot(time, activation, label="activation")

        plt.xlabel("Time [s]")
        plt.ylabel("Value")
        plt.title(f"Muscle: {muscle_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(save_path)
        plt.close()

        # print(f"[MusclePlotter] wrote {save_path}")

    def _extract_series(self, values, index):
        """
        stepごとの配列から指定筋indexの時系列を取り出す。
        """

        series = []

        for row in values:
            if row is None or len(row) <= index:
                series.append(np.nan)
            else:
                series.append(float(row[index]))

        return np.asarray(series, dtype=float)