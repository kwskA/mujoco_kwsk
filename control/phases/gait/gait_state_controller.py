# control/gait_state_controller.py

from dataclasses import dataclass
import mujoco


class GaitState:
    UNKNOWN = "Unknown"
    EARLY_STANCE = "EarlyStance"
    LATE_STANCE = "LateStance"
    LIFTOFF = "Liftoff"
    SWING = "Swing"
    LANDING = "Landing"


@dataclass
class LegState:
    side: str
    foot_body_name: str
    toe_body_name: str

    state: str = GaitState.UNKNOWN

    foot_contact: bool = False
    toe_contact: bool = False
    contact: bool = False

    sagittal_pos: float = 0.0


class GaitStateController:
    """
    MuJoCo用の歩行相判定器。

    判定に使うもの：
    - 踵側 body の接地
    - つま先 body の接地
    - 足部位置 foot_x - pelvis_x
    - 反対脚との前後関係
    """

    def __init__(
        self,
        model,
        pelvis_body_name="pelvis",
        right_foot_body_name="calcn_r",
        left_foot_body_name="calcn_l",
        right_toe_body_name="toes_r",
        left_toe_body_name="toes_l",
        ground_geom_name="ground-plane",
        late_stance_threshold=-0.05,
        liftoff_threshold=-0.15,
        landing_threshold=0.05,
    ):
        self.model = model

        self.pelvis_body_id = self._get_body_id(pelvis_body_name)
        self.ground_geom_id = self._get_geom_id(ground_geom_name)

        self.right = LegState(
            side="r",
            foot_body_name=right_foot_body_name,
            toe_body_name=right_toe_body_name,
        )

        self.left = LegState(
            side="l",
            foot_body_name=left_foot_body_name,
            toe_body_name=left_toe_body_name,
        )

        self.right_foot_body_id = self._get_body_id(right_foot_body_name)
        self.left_foot_body_id = self._get_body_id(left_foot_body_name)

        self.right_toe_body_id = self._get_body_id(right_toe_body_name)
        self.left_toe_body_id = self._get_body_id(left_toe_body_name)

        self.late_stance_threshold = late_stance_threshold
        self.liftoff_threshold = liftoff_threshold
        self.landing_threshold = landing_threshold

    def _get_body_id(self, body_name):
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )

        if body_id == -1:
            raise ValueError(f"Body not found: {body_name}")

        return body_id

    def _get_geom_id(self, geom_name):
        geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name,
        )

        if geom_id == -1:
            raise ValueError(f"Geom not found: {geom_name}")

        return geom_id

    def update(self, data):
        """
        毎step呼ぶ。
        戻り値：
            state_r, state_l
        """

        pelvis_x = data.xpos[self.pelvis_body_id][0]

        self.right.foot_contact = self._is_body_contact_with_ground(
            data,
            self.right_foot_body_id,
        )
        self.right.toe_contact = self._is_body_contact_with_ground(
            data,
            self.right_toe_body_id,
        )
        self.right.contact = self.right.foot_contact or self.right.toe_contact

        self.left.foot_contact = self._is_body_contact_with_ground(
            data,
            self.left_foot_body_id,
        )
        self.left.toe_contact = self._is_body_contact_with_ground(
            data,
            self.left_toe_body_id,
        )
        self.left.contact = self.left.foot_contact or self.left.toe_contact

        self.right.sagittal_pos = data.xpos[self.right_foot_body_id][0] - pelvis_x
        self.left.sagittal_pos = data.xpos[self.left_foot_body_id][0] - pelvis_x

        self.right.state = self._update_one_leg(
            leg=self.right,
            opposite_leg=self.left,
        )

        self.left.state = self._update_one_leg(
            leg=self.left,
            opposite_leg=self.right,
        )

        return self.right.state, self.left.state

    def _is_body_contact_with_ground(self, data, body_id):
        """
        指定bodyに属するgeomが ground-plane と接触しているか判定。
        """

        for i in range(data.ncon):
            contact = data.contact[i]

            geom1 = contact.geom1
            geom2 = contact.geom2

            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]

            geom1_is_ground = geom1 == self.ground_geom_id
            geom2_is_ground = geom2 == self.ground_geom_id

            body1_is_target = body1 == body_id
            body2_is_target = body2 == body_id

            if (geom1_is_ground and body2_is_target) or (
                geom2_is_ground and body1_is_target
            ):
                return True

        return False

    def _update_one_leg(self, leg, opposite_leg):
        """
        5状態への簡易遷移。
        """

        old_state = leg.state

        contact = leg.contact
        opposite_contact = opposite_leg.contact

        sag = leg.sagittal_pos
        opposite_sag = opposite_leg.sagittal_pos

        if old_state == GaitState.UNKNOWN:
            if contact:
                if sag < self.late_stance_threshold:
                    return GaitState.LATE_STANCE
                return GaitState.EARLY_STANCE
            else:
                if sag > self.landing_threshold:
                    return GaitState.LANDING
                return GaitState.SWING

        if old_state == GaitState.EARLY_STANCE:
            if not contact:
                return GaitState.SWING
            if opposite_contact and sag < opposite_sag:
                return GaitState.LIFTOFF
            if sag < self.late_stance_threshold:
                return GaitState.LATE_STANCE
            return GaitState.EARLY_STANCE

        if old_state == GaitState.LATE_STANCE:
            if not contact:
                return GaitState.SWING
            if opposite_contact and sag < opposite_sag:
                return GaitState.LIFTOFF
            if sag < self.liftoff_threshold:
                return GaitState.LIFTOFF
            return GaitState.LATE_STANCE

        if old_state == GaitState.LIFTOFF:
            if not contact:
                return GaitState.SWING
            return GaitState.LIFTOFF

        if old_state == GaitState.SWING:
            if contact:
                return GaitState.EARLY_STANCE
            if sag > self.landing_threshold:
                return GaitState.LANDING
            return GaitState.SWING

        if old_state == GaitState.LANDING:
            if contact:
                return GaitState.EARLY_STANCE
            return GaitState.LANDING

        return old_state

    def get_debug_info(self):
        return {
            "right_state": self.right.state,
            "left_state": self.left.state,

            "right_contact": self.right.contact,
            "left_contact": self.left.contact,

            "right_heel_contact": self.right.foot_contact,
            "right_toe_contact": self.right.toe_contact,
            "left_heel_contact": self.left.foot_contact,
            "left_toe_contact": self.left.toe_contact,

            "right_sagittal_pos": self.right.sagittal_pos,
            "left_sagittal_pos": self.left.sagittal_pos,
        }