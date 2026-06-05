# optim/initial_state_writer.py

import os
import csv
import json
import numpy as np
from mujoco import mj_forward


class InitialStateWriter:
    """
    最適化開始時の初期姿勢・初期筋状態を保存するクラス。
    """

    def __init__(
        self,
        save_dir,
        model,
        data,
        muscle_names,
        model_path=None,
        extra_info=None,
    ):
        self.save_dir = save_dir
        self.model = model
        self.data = data
        self.muscle_names = muscle_names
        self.model_path = model_path
        self.extra_info = extra_info if extra_info is not None else {}

        os.makedirs(self.save_dir, exist_ok=True)

    def write_all(self):
        """
        初期状態の情報をまとめて保存する。
        """

        mj_forward(self.model, self.data)

        self.write_qpos()
        self.write_muscle_state()
        self.write_info()

    def write_qpos(self):
        """
        初期qposを関節名付きCSVで保存する。
        """

        path = os.path.join(self.save_dir, "initial_qpos.csv")

        qpos_names = self._get_qpos_names()

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "name", "value"])

            for i, value in enumerate(self.data.qpos):
                writer.writerow([
                    i,
                    qpos_names[i],
                    float(value),
                ])

        print(f"[InitialStateWriter] wrote {path}")

    def write_muscle_state(self):
        """
        初期筋長・筋速度・筋力・activationをCSVで保存する。
        """

        path = os.path.join(self.save_dir, "initial_muscle_state.csv")

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "muscle",
                "tendon_id",
                "actuator_id",
                "initial_length",
                "initial_velocity",
                "initial_force",
                "initial_activation",
            ])

            for muscle in self.muscle_names:
                tendon_id = self.model.tendon(f"{muscle}_tendon").id
                actuator_id = self.model.actuator(muscle).id

                writer.writerow([
                    muscle,
                    tendon_id,
                    actuator_id,
                    float(self.data.ten_length[tendon_id]),
                    float(self.data.ten_velocity[tendon_id]),
                    float(self.data.actuator_force[actuator_id]),
                    float(self.data.ctrl[actuator_id]),
                ])

        print(f"[InitialStateWriter] wrote {path}")

    def write_info(self):
        """
        初期状態に関する補足情報をJSONで保存する。
        """

        path = os.path.join(self.save_dir, "initial_info.json")

        info = {
            "model_path": self.model_path,

            "nq": int(self.model.nq),
            "nv": int(self.model.nv),
            "nu": int(self.model.nu),
            "nbody": int(self.model.nbody),

            "num_muscles": len(self.muscle_names),
            "muscle_names": list(self.muscle_names),

            "initial_com": self.data.subtree_com[0].tolist(),
            "initial_com_x": float(self.data.subtree_com[0][0]),
            "initial_com_y": float(self.data.subtree_com[0][1]),
            "initial_com_z": float(self.data.subtree_com[0][2]),
        }

        info.update(self.extra_info)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

        print(f"[InitialStateWriter] wrote {path}")

    def _get_qpos_names(self):
        """
        qpos各要素に対応する名前を作成する。
        """

        qpos_names = [None for _ in range(self.model.nq)]

        for joint_id in range(self.model.njnt):
            joint_name = self.model.joint(joint_id).name
            qpos_adr = self.model.jnt_qposadr[joint_id]
            qpos_dim = self._get_joint_qpos_dim(joint_id)

            if qpos_dim == 1:
                qpos_names[qpos_adr] = joint_name
            else:
                for k in range(qpos_dim):
                    qpos_names[qpos_adr + k] = f"{joint_name}_{k}"

        for i, name in enumerate(qpos_names):
            if name is None:
                qpos_names[i] = f"qpos_{i}"

        return qpos_names

    def _get_joint_qpos_dim(self, joint_id):
        """
        joint typeからqpos次元数を返す。
        """

        joint_type = self.model.jnt_type[joint_id]

        if joint_type == 0:
            return 7

        if joint_type == 1:
            return 4

        return 1