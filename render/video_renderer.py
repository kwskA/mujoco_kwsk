# render/video_renderer.py

import os
import numpy as np
import skvideo.io
import mujoco

from mujoco import MjData, mj_step, mj_forward


class VideoRenderer:
    """
    MuJoCoシミュレーションを動画として保存するクラス。

    controllerで制御入力を計算しながら再シミュレーションし、
    tracking_body_nameで指定したbodyを追従して描画する。
    """

    def __init__(
        self,
        model,
        controller,
        sim_steps,
        initial_qpos=None,
        width=400,
        height=400,
        fps=200,
        tracking_body_name="pelvis",
        camera_distance=3.0,
        camera_azimuth=90.0,
        camera_elevation=-15.0,
    ):
        """
        動画出力に必要なモデル，制御器，描画設定を保存する。
        """

        self.model = model
        self.controller = controller
        self.sim_steps = int(sim_steps)
        self.initial_qpos = initial_qpos

        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self.tracking_body_name = tracking_body_name
        self.camera_distance = float(camera_distance)
        self.camera_azimuth = float(camera_azimuth)
        self.camera_elevation = float(camera_elevation)

        self.tracking_body_id = None

        if self.tracking_body_name is not None:
            self.tracking_body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.tracking_body_name,
            )

            if self.tracking_body_id == -1:
                raise ValueError(
                    f"tracking body not found: {self.tracking_body_name}"
                )

    def reset_data(self, data):
        """
        MjDataを初期状態に戻す。
        """

        if self.initial_qpos is not None:
            data.qpos[:] = self.initial_qpos.copy()
        else:
            data.qpos[:] = self.model.key_qpos[0].copy()

        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.ctrl[:] = 0.0

        mj_forward(self.model, data)

    def _ensure_dir(self, path):
        """
        保存先ディレクトリを作成する。
        """

        os.makedirs(path, exist_ok=True)

    def _create_tracking_camera(self):
        """
        追従カメラを作成する。
        """

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        cam.distance = self.camera_distance
        cam.azimuth = self.camera_azimuth
        cam.elevation = self.camera_elevation

        return cam

    def _update_tracking_camera(self, cam, data):
        """
        tracking_bodyの位置にカメラの注視点を更新する。
        """

        if self.tracking_body_id is None:
            return

        target_pos = data.xpos[self.tracking_body_id]
        cam.lookat[:] = target_pos

    def render(
        self,
        save_path,
        tendon_ids=None,
        actuator_ids=None,
        color_tendons=True,
    ):
        """
        シミュレーションを実行しながら動画を保存する。
        """

        save_dir = os.path.dirname(save_path)

        if save_dir:
            self._ensure_dir(save_dir)

        data = MjData(self.model)
        self.reset_data(data)

        # print("[VideoRenderer] initial_qpos[:8] =", self.initial_qpos[:8])
        # print("[VideoRenderer] reset qpos[:8] =", data.qpos[:8])
        # print("[VideoRenderer] reset com =", data.subtree_com[0])

        renderer = mujoco.Renderer(
            self.model,
            height=self.height,
            width=self.width,
        )

        tracking_camera = self._create_tracking_camera()

        video_writer = skvideo.io.FFmpegWriter(
            save_path,
            inputdict={
                "-r": str(self.fps),
            },
            outputdict={
                "-pix_fmt": "yuv420p",
            },
        )

        try:
            for step in range(self.sim_steps):

                ctrl = self.controller.compute_ctrl(data)

                data.ctrl[:] = ctrl

                mj_step(self.model, data)

                if (
                    color_tendons
                    and tendon_ids is not None
                    and actuator_ids is not None
                ):
                    self._color_tendons(
                        tendon_ids=tendon_ids,
                        actuator_ids=actuator_ids,
                        ctrl=ctrl,
                    )

                self._update_tracking_camera(
                    tracking_camera,
                    data,
                )

                renderer.update_scene(
                    data,
                    camera=tracking_camera,
                )

                frame = renderer.render()
                video_writer.writeFrame(frame)

        finally:
            try:
                video_writer.close()
            except Exception as e:
                print(f"[VideoRenderer] video_writer close skipped: {e}")

        print(f"[VideoRenderer] wrote {save_path}")

    def _color_tendons(
        self,
        tendon_ids,
        actuator_ids,
        ctrl,
    ):
        """
        筋activationに応じて腱の色を変更する。
        """

        for tendon_id, actuator_id in zip(tendon_ids, actuator_ids):

            activation = float(ctrl[actuator_id])
            activation = np.clip(activation, 0.0, 1.0)

            self.model.tendon_rgba[tendon_id] = np.array([
                activation,
                0.0,
                1.0 - activation,
                1.0,
            ])