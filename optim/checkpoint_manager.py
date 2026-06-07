# optim/checkpoint_manager.py

import os
import json
import numpy as np
from mujoco import MjModel, MjData, mj_forward

from simulation.runner import SimulationRunner
from simulation.logs.csv_log_writer import CSVLogWriter
from render.video_renderer import VideoRenderer
from render.plot_objectives import ObjectivePlotter
from render.plot_muscles import MusclePlotter


class CheckpointManager:
    """
    最適化途中・終了時の最良解を保存するクラス。
    """

    def __init__(
        self,
        result_dir,
        model_path,
        controller_builder,
        objective_manager_builder,
        sim_steps,
        initial_qpos=None,
        muscle_names=None,
    ):
        self.result_dir = result_dir
        self.model_path = model_path
        self.controller_builder = controller_builder
        self.objective_manager_builder = objective_manager_builder
        self.sim_steps = int(sim_steps)
        self.initial_qpos = initial_qpos
        self.muscle_names = muscle_names

        os.makedirs(self.result_dir, exist_ok=True)

    def save_checkpoint(
        self,
        generation,
        best_cost,
        best_params,
        tag=None,
        write_video=True,
        write_plots=True,
        write_csv=True,
    ):
        """
        best_paramsを使って再シミュレーションし，結果を保存する。
        """

        if tag is None:
            tag = f"gen_{generation:04d}"

        checkpoint_dir = os.path.join(
            self.result_dir,
            "checkpoints",
            tag,
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        np.save(
            os.path.join(checkpoint_dir, "best_params.npy"),
            np.asarray(best_params, dtype=float),
        )

        info = {
            "generation": int(generation),
            "best_cost": float(best_cost),
            "tag": tag,
        }

        with open(
            os.path.join(checkpoint_dir, "checkpoint_info.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

        model = MjModel.from_xml_path(self.model_path)
        data = MjData(model)

        if self.initial_qpos is not None:
            data.qpos[:] = self.initial_qpos.copy()
        else:
            data.qpos[:] = model.key_qpos[0].copy()

        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.ctrl[:] = 0.0

        mj_forward(model, data)

        initial_qpos_for_simulation = data.qpos.copy()

        qpos_names = self._get_qpos_names(model)

        # self._save_initial_qpos(
        #     checkpoint_dir=checkpoint_dir,
        #     qpos_names=qpos_names,
        #     initial_qpos=initial_qpos_for_simulation,
        # )

        controller = self.controller_builder(model)
        controller.set_params_from_vector(best_params)

        objective_manager = self.objective_manager_builder()

        tendon_ids = None
        actuator_ids = None

        if self.muscle_names is not None:
            tendon_ids = np.array([
                model.tendon(f"{m}_tendon").id
                for m in self.muscle_names
            ], dtype=int)

            actuator_ids = np.array([
                model.actuator(m).id
                for m in self.muscle_names
            ], dtype=int)

        qvel_names = [
            f"dof_{i}"
            for i in range(model.nv)
        ]

        runner = SimulationRunner(
            model=model,
            controller=controller,
            objective_manager=objective_manager,
            sim_steps=self.sim_steps,
            initial_qpos=initial_qpos_for_simulation,
            qpos_names=qpos_names,
            qvel_names=qvel_names,
            muscle_names=self.muscle_names,
            tendon_ids=tendon_ids,
            actuator_ids=actuator_ids,
            enable_log=True,
        )

        result = runner.run(data)
        sim_log = result["simulation_log"]

        if write_csv:
            csv_dir = os.path.join(checkpoint_dir, "csv")
            CSVLogWriter(csv_dir).write_all(sim_log)

        if write_plots:
            plot_dir = os.path.join(checkpoint_dir, "plots")

            ObjectivePlotter(
                os.path.join(plot_dir, "objectives")
            ).plot(sim_log)

            MusclePlotter(
                os.path.join(plot_dir, "muscles")
            ).plot_all(sim_log)

        if write_video:
            video_dir = os.path.join(checkpoint_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)

            camera_settings = {
                "front": {
                    "azimuth": 90.0,
                    "elevation": -10.0,
                    "distance": 3.0,
                },
                "side": {
                    "azimuth": 0.0,
                    "elevation": -10.0,
                    "distance": 3.0,
                },
                "diagonal": {
                    "azimuth": 45.0,
                    "elevation": -15.0,
                    "distance": 3.0,
                },
                "oblique": {
                    "azimuth": 135.0,
                    "elevation": -20.0,
                    "distance": 3.2,
                },
            }

            for camera_name, camera_param in camera_settings.items():

                video_path = os.path.join(
                    video_dir,
                    f"{tag}_{camera_name}.mp4",
                )

                video_renderer = VideoRenderer(
                    model=model,
                    controller=controller,
                    sim_steps=self.sim_steps,
                    initial_qpos=initial_qpos_for_simulation,
                    tracking_body_name="pelvis",
                    camera_distance=camera_param["distance"],
                    camera_azimuth=camera_param["azimuth"],
                    camera_elevation=camera_param["elevation"],
                )

                video_renderer.render(
                    save_path=video_path,
                    tendon_ids=tendon_ids,
                    actuator_ids=actuator_ids,
                    color_tendons=True,
                )

        print(f"[CheckpointManager] saved {checkpoint_dir}")

    def _get_qpos_names(self, model):
        """
        qposの各要素に対応する名前を作成する。
        """

        qpos_names = [
            None
            for _ in range(model.nq)
        ]

        for joint_id in range(model.njnt):
            joint_name = model.joint(joint_id).name
            qpos_adr = model.jnt_qposadr[joint_id]
            qpos_dim = self._get_joint_qpos_dim(model, joint_id)

            if qpos_dim == 1:
                qpos_names[qpos_adr] = joint_name
            else:
                for k in range(qpos_dim):
                    qpos_names[qpos_adr + k] = f"{joint_name}_{k}"

        for i, name in enumerate(qpos_names):
            if name is None:
                qpos_names[i] = f"qpos_{i}"

        return qpos_names

    def _get_joint_qpos_dim(self, model, joint_id):
        """
        joint typeからqpos次元数を返す。
        """

        joint_type = model.jnt_type[joint_id]

        # free joint
        if joint_type == 0:
            return 7

        # ball joint
        if joint_type == 1:
            return 4

        # slide / hinge joint
        return 1

    # def _save_initial_qpos(
    #     self,
    #     checkpoint_dir,
    #     qpos_names,
    #     initial_qpos,
    # ):
    #     """
    #     checkpointで使用した初期姿勢qposをCSVとして保存する。
    #     """

    #     path = os.path.join(
    #         checkpoint_dir,
    #         "initial_qpos.csv",
    #     )

    #     with open(path, "w", newline="", encoding="utf-8") as f:
    #         writer = csv.writer(f)

    #         writer.writerow([
    #             "index",
    #             "name",
    #             "value",
    #         ])

    #         for i, value in enumerate(initial_qpos):
    #             writer.writerow([
    #                 i,
    #                 qpos_names[i],
    #                 float(value),
    #             ])

    #     print(f"[CheckpointManager] wrote {path}")
