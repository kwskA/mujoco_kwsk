# simulation/runner.py

from mujoco import MjData, mj_step, mj_forward
from simulation.logs.simulation_log import SimulationLog


class SimulationRunner:
    """
    MuJoCoシミュレーションを1回実行するクラス。

    controllerで制御入力を計算し，
    objective_managerで評価関数を更新する。
    必要な場合だけSimulationLogを作成する。
    """

    def __init__(
        self,
        model,
        controller,
        objective_manager,
        sim_steps,
        initial_qpos=None,
        initial_qvel=None,
        fall_detector=None,
        qpos_names=None,
        qvel_names=None,
        muscle_names=None,
        tendon_ids=None,
        actuator_ids=None,
        enable_log=False,
    ):
        """
        シミュレーションに必要なモデル，制御器，評価関数群を保存する。
        """

        self.model = model
        self.controller = controller
        self.objective_manager = objective_manager
        self.sim_steps = int(sim_steps)

        self.initial_qpos = initial_qpos
        self.initial_qvel = initial_qvel
        self.fall_detector = fall_detector

        self.qpos_names = qpos_names
        self.qvel_names = qvel_names
        self.muscle_names = muscle_names
        self.tendon_ids = tendon_ids
        self.actuator_ids = actuator_ids

        self.enable_log = bool(enable_log)

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

        # jid = self.model.joint("pelvis_tx").id
        # adr = self.model.jnt_dofadr[jid]
        # print("[reset_data] pelvis_tx qvel =", data.qvel[adr])

        mj_forward(self.model, data)

        if hasattr(self.controller, "reset"):
            self.controller.reset()

    def _get_phase_name(self, data):
        """
        現在の動作フェーズ名を取得する。
        """

        if hasattr(self.controller, "get_current_phase"):
            phase = self.controller.get_current_phase(data.time)

            if hasattr(phase, "name"):
                return phase.name

            return str(phase)

        if hasattr(self.controller, "name"):
            return self.controller.name

        return None

    def _get_target_length(self, data):
        """
        現在フェーズの制御器からtarget_lengthを取得する。
        ログ出力時のみ使用する。
        """

        controller = self.controller

        if hasattr(controller, "get_current_phase"):
            phase = controller.get_current_phase(data.time)

            if hasattr(phase, "control_method"):
                method = phase.control_method

                if hasattr(method, "target_length"):
                    return method.target_length

        if hasattr(controller, "control_method"):
            method = controller.control_method

            if hasattr(method, "target_length"):
                return method.target_length

        if hasattr(controller, "target_length"):
            return controller.target_length

        return None

    def _create_simulation_log(self):
        """
        SimulationLogを作成する。
        """

        objective_names = [
            objective.name
            for objective in self.objective_manager.objectives
        ]

        return SimulationLog(
            qpos_names=self.qpos_names,
            qvel_names=self.qvel_names,
            muscle_names=self.muscle_names,
            objective_names=objective_names,
        )

    def _record_log_step(
        self,
        sim_log,
        step,
        data,
        ctrl,
        phase_name,
    ):
        """
        1step分のログをSimulationLogへ記録する。
        """

        _, objective_details = self.objective_manager.finalize()

        tendon_length = None
        tendon_velocity = None
        actuator_force = None
        activation = None

        if self.tendon_ids is not None:
            tendon_length = data.ten_length[self.tendon_ids]
            tendon_velocity = data.ten_velocity[self.tendon_ids]

        if self.actuator_ids is not None:
            actuator_force = data.actuator_force[self.actuator_ids]
            activation = ctrl[self.actuator_ids]

        target_length = self._get_target_length(data)

        sim_log.record_step(
            step=step,
            time=data.time,
            phase_name=phase_name,
            qpos=data.qpos,
            qvel=data.qvel,
            com=data.subtree_com[0],
            tendon_length=tendon_length,
            tendon_velocity=tendon_velocity,
            actuator_force=actuator_force,
            activation=activation,
            objective_values=objective_details,
            target_length=target_length,
        )

    def run(self, data=None):
        """
        シミュレーションを実行し，実行結果を返す。
        """

        if data is None:
            data = MjData(self.model)

        self.reset_data(data)

        # print("[before step] time =", data.time)
        # print("[before step] pelvis_tx =", data.qpos[0])
        # print("[before step] pelvis_tx qvel =", data.qvel[0])

        self.objective_manager.reset()

        sim_log = None

        if self.enable_log:
            sim_log = self._create_simulation_log()

        survived_steps = self.sim_steps
        fallen = False

        for step in range(self.sim_steps):

            phase_name = self._get_phase_name(data)

            ctrl = self.controller.compute_ctrl(data)

            data.ctrl[:] = ctrl

            mj_step(self.model, data)

            self.objective_manager.update(
                step=step,
                time=data.time,
                model=self.model,
                data=data,
                ctrl=ctrl,
                phase_name=phase_name,
            )

            if self.enable_log:
                self._record_log_step(
                    sim_log=sim_log,
                    step=step,
                    data=data,
                    ctrl=ctrl,
                    phase_name=phase_name,
                )

            if self.fall_detector is not None:
                if self.fall_detector.is_fallen(
                    model=self.model,
                    data=data,
                    step=step,
                ):
                    survived_steps = step
                    fallen = True
                    break

        return {
            "data": data,
            "survived_steps": survived_steps,
            "sim_steps": self.sim_steps,
            "fallen": fallen,
            "simulation_log": sim_log,
        }