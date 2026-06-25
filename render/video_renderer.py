# render/video_renderer.py

import os
import numpy as np
import skvideo.io
import mujoco

from mujoco import MjData, mj_step, mj_forward


CAMERA_PRESETS = {
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


class VideoRenderer:
    """
    MuJoCoシミュレーションを動画として保存するクラス。

    controllerで制御入力を計算しながら再シミュレーションし、
    tracking_body_nameで指定したbodyを追従して描画する。

    XML側にmocap bodyとして以下を用意しておくと，
    COMとCOPを動画中に表示できる。

    - com_marker
    - cop_marker
    """

    def __init__(
        self,
        model,
        controller,
        sim_steps,
        initial_qpos=None,
        initial_qvel=None,
        width=400,
        height=400,
        fps=200,
        tracking_body_name="pelvis",
        camera_distance=3.0,
        camera_azimuth=90.0,
        camera_elevation=-15.0,
        com_marker_body_name="com_marker",
        cop_marker_body_name="cop_marker",
        show_com=True,
        show_cop=True,
        vertical_axis=1,
    ):
        """
        動画出力に必要なモデル，制御器，描画設定を保存する。

        vertical_axis
        -------------
        鉛直方向の軸。
        このモデルでは pelvis_ty が上下方向なので，通常は 1 を使う。

        0 : X軸
        1 : Y軸
        2 : Z軸
        """

        self.model = model
        self.controller = controller
        self.sim_steps = int(sim_steps)
        self.initial_qpos = initial_qpos
        self.initial_qvel = initial_qvel

        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self.tracking_body_name = tracking_body_name
        self.camera_distance = float(camera_distance)
        self.camera_azimuth = float(camera_azimuth)
        self.camera_elevation = float(camera_elevation)

        self.com_marker_body_name = com_marker_body_name
        self.cop_marker_body_name = cop_marker_body_name

        self.show_com = bool(show_com)
        self.show_cop = bool(show_cop)

        self.vertical_axis = int(vertical_axis)

        if self.vertical_axis not in [0, 1, 2]:
            raise ValueError(
                "vertical_axis must be 0, 1, or 2. "
                f"got {self.vertical_axis}"
            )

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

        self.com_marker_body_id = self._get_body_id(
            self.com_marker_body_name
        )

        self.cop_marker_body_id = self._get_body_id(
            self.cop_marker_body_name
        )

        self.com_mocap_id = self._get_mocap_id(
            body_id=self.com_marker_body_id,
            body_name=self.com_marker_body_name,
        )

        self.cop_mocap_id = self._get_mocap_id(
            body_id=self.cop_marker_body_id,
            body_name=self.cop_marker_body_name,
        )

        self.last_cop_pos = None

    def _get_body_id(self, body_name):
        """
        body名からbody idを取得する。
        存在しない場合はNoneを返す。
        """

        if body_name is None:
            return None

        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )

        if body_id == -1:
            return None

        return body_id

    def _get_mocap_id(
        self,
        body_id,
        body_name,
    ):
        """
        body idからmocap idを取得する。
        bodyが存在しない場合，またはmocap bodyでない場合はNoneを返す。
        """

        if body_id is None:
            return None

        mocap_id = int(self.model.body_mocapid[body_id])

        if mocap_id == -1:
            print(
                f"[VideoRenderer] warning: body '{body_name}' exists, "
                "but it is not a mocap body. "
                "COM/COP marker will not be updated."
            )
            return None

        return mocap_id

    def reset_data(self, data):
        """
        MjDataを初期状態に戻す。
        """

        if self.initial_qpos is not None:
            data.qpos[:] = self.initial_qpos.copy()
        else:
            data.qpos[:] = self.model.key_qpos[0].copy()

        if self.initial_qvel is not None:
            data.qvel[:] = self.initial_qvel.copy()
        else:
            data.qvel[:] = 0.0
            
        data.qacc[:] = 0.0
        data.ctrl[:] = 0.0

        mj_forward(self.model, data)

        self.last_cop_pos = None

        self._update_com_marker(data)

        initial_cop = self._compute_cop(data)

        if initial_cop is None:
            initial_cop = data.subtree_com[0].copy()
            initial_cop[self.vertical_axis] = 0.0

        self.last_cop_pos = initial_cop.copy()

        if self.cop_mocap_id is not None:
            data.mocap_pos[self.cop_mocap_id] = initial_cop

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

    def _update_com_marker(self, data):
        """
        COMマーカーを全身COM位置に移動する。
        """

        if not self.show_com:
            return

        if self.com_mocap_id is None:
            return

        com_pos = data.subtree_com[0].copy()

        data.mocap_pos[self.com_mocap_id] = com_pos

    def _compute_cop(self, data):
        """
        接触点と接触法線力からCOPを計算する。

        MuJoCoの mj_contactForce で得られる force_local[0] は
        接触法線方向の力なので，これを重みとして接触位置を平均する。
        """

        weighted_pos = np.zeros(3)
        total_normal_force = 0.0

        ground_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "ground-plane",
        )

        for i in range(data.ncon):
            contact = data.contact[i]

            # 床との接触のみ使う
            if ground_geom_id != -1:
                if (
                    contact.geom1 != ground_geom_id
                    and contact.geom2 != ground_geom_id
                ):
                    continue

            force_local = np.zeros(6)

            mujoco.mj_contactForce(
                self.model,
                data,
                i,
                force_local,
            )

            # force_local[0] が接触法線方向の力
            normal_force = force_local[0]

            if normal_force <= 1.0e-8:
                continue

            contact_pos = contact.pos.copy()

            weighted_pos += normal_force * contact_pos
            total_normal_force += normal_force

        if total_normal_force <= 1.0e-8:
            return None

        cop_pos = weighted_pos / total_normal_force

        # このモデルはY軸が鉛直方向なので，COPを床面上に固定
        cop_pos[self.vertical_axis] = 0.0

        return cop_pos

    def _update_cop_marker(self, data):
        """
        COPマーカーをCOP位置に移動する。

        接触がない場合は直前のCOP位置を保持する。
        """

        if not self.show_cop:
            return

        if self.cop_mocap_id is None:
            return

        cop_pos = self._compute_cop(data)

        if cop_pos is None:
            if self.last_cop_pos is None:
                return

            cop_pos = self.last_cop_pos.copy()
        else:
            self.last_cop_pos = cop_pos.copy()

        data.mocap_pos[self.cop_mocap_id] = cop_pos

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

                self._update_com_marker(data)
                self._update_cop_marker(data)

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