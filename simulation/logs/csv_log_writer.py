# simulation/logs/csv_log_writer.py

import os
import csv
import numpy as np


class CSVLogWriter:
    """
    SimulationLogの内容をCSVファイルとして保存するクラス。
    """

    def __init__(self, save_dir):
        """
        CSVファイルの保存先ディレクトリを設定する。
        """

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def write_all(self, sim_log):
        """
        kinematics, muscles, objectives のCSVをまとめて保存する。
        """

        self.write_kinematics(sim_log)
        self.write_muscles(sim_log)
        self.write_objectives(sim_log)

    def write_kinematics(self, sim_log):
        """
        qpos, qvel, COM, COP, GRF, phaseをkinematics.csvに保存する。
        """

        path = os.path.join(self.save_dir, "kinematics.csv")

        qpos_names = sim_log.qpos_names
        qvel_names = sim_log.qvel_names

        qpos_columns = [
            f"qpos_{name}" if name is not None else f"qpos_{i}"
            for i, name in enumerate(qpos_names)
        ]

        qvel_columns = [
            f"qvel_{name}" if name is not None else f"qvel_{i}"
            for i, name in enumerate(qvel_names)
        ]

        header = (
            ["step", "time", "phase_name"]
            + qpos_columns
            + qvel_columns
            + ["com_x", "com_y", "com_z"]
            + ["cop_x", "cop_y", "cop_z"]
            + ["grf_x", "grf_y", "grf_z"]
        )

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for i in range(sim_log.get_length()):
                row = [
                    sim_log.step[i],
                    sim_log.time[i],
                    sim_log.phase_name[i],
                ]

                row.extend(self._to_list(sim_log.qpos[i]))
                row.extend(self._to_list(sim_log.qvel[i]))
                row.extend(self._to_list(sim_log.com[i]))
                row.extend(self._to_list(sim_log.cop[i]))
                row.extend(self._to_list(sim_log.grf[i]))

                writer.writerow(row)

    def write_muscles(self, sim_log):
        """
        筋長, 筋速度, 筋力, activationをmuscles.csvに保存する。
        """

        path = os.path.join(self.save_dir, "muscles.csv")

        muscle_names = sim_log.muscle_names

        header = ["step", "time", "phase_name"]

        for name in muscle_names:
            header.append(f"{name}.tendon_length")
            header.append(f"{name}.tendon_velocity")
            header.append(f"{name}.actuator_force")
            header.append(f"{name}.activation")

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for i in range(sim_log.get_length()):
                row = [
                    sim_log.step[i],
                    sim_log.time[i],
                    sim_log.phase_name[i],
                ]

                tendon_length = self._to_list(sim_log.tendon_length[i])
                tendon_velocity = self._to_list(sim_log.tendon_velocity[i])
                actuator_force = self._to_list(sim_log.actuator_force[i])
                activation = self._to_list(sim_log.activation[i])

                for m_idx in range(len(muscle_names)):
                    row.append(self._safe_get(tendon_length, m_idx))
                    row.append(self._safe_get(tendon_velocity, m_idx))
                    row.append(self._safe_get(actuator_force, m_idx))
                    row.append(self._safe_get(activation, m_idx))

                writer.writerow(row)

    def write_objectives(self, sim_log):
        """
        各評価関数の値をobjectives.csvに保存する。
        """

        path = os.path.join(self.save_dir, "objectives.csv")

        objective_names = sim_log.objective_names

        header = (
            ["step", "time", "phase_name"]
            + objective_names
        )

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for i in range(sim_log.get_length()):
                objective_values = sim_log.objective_values[i]

                row = [
                    sim_log.step[i],
                    sim_log.time[i],
                    sim_log.phase_name[i],
                ]

                for name in objective_names:
                    row.append(
                        objective_values.get(name, np.nan)
                    )

                writer.writerow(row)

    def _to_list(self, value):
        """
        numpy配列などをlistへ変換する。
        """

        if value is None:
            return []

        return np.asarray(value, dtype=float).tolist()

    def _safe_get(self, values, index):
        """
        listの範囲外アクセスをNaNで補う。
        """

        if index >= len(values):
            return np.nan

        return values[index]