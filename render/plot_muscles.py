# render/plot_muscles.py

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


class MusclePlotter:
    """
    各筋について target_length, actual_length, activation をグラフ化するクラス。
    1次元・2次元・それ以上のログにもできるだけ安全に対応する。
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

    def _extract_series(self, values, index):
        """
        stepごとの配列から指定筋indexの時系列を取り出す。

        対応例：
        - row shape = (num_muscles,)
            -> row[index]

        - row shape = (num_states, num_muscles)
            -> row[:, index] の平均値

        - row shape = (num_muscles, num_states)
            -> row[index, :] の平均値

        - rowがスカラー
            -> その値

        どの形でも、最終的には各stepにつき1つのfloatへ変換する。
        """

        series = []

        for row in values:
            value = self._extract_value_from_row(
                row=row,
                index=index,
            )
            series.append(value)

        return np.asarray(series, dtype=float)

    def _extract_value_from_row(self, row, index):
        """
        1step分のログから指定筋indexに対応する代表値を取り出す。
        """

        if row is None:
            return np.nan

        arr = np.asarray(row, dtype=float)

        if arr.size == 0:
            return np.nan

        if arr.ndim == 0:
            return float(arr)

        if arr.ndim == 1:
            if arr.shape[0] <= index:
                return np.nan

            return float(arr[index])

        if arr.ndim == 2:
            # shape = (num_states, num_muscles)
            if arr.shape[1] > index:
                values = arr[:, index]
                return float(np.nanmean(values))

            # shape = (num_muscles, num_states)
            if arr.shape[0] > index:
                values = arr[index, :]
                return float(np.nanmean(values))

            return np.nan

        # 3次元以上の場合は flatten して index を使う
        flat = arr.reshape(-1)

        if flat.shape[0] <= index:
            return np.nan

        return float(flat[index])