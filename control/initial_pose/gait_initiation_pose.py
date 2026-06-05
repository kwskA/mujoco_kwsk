# control/initial_pose/gait_initiation_pose.py

from mujoco import mj_forward


GAIT_INITIATION_INITIAL_POSE = {
    "pelvis_tx": 0.0,
    "pelvis_ty": 0.900237,
    "pelvis_tilt": -0.105763,

    "hip_flexion_r": 0.439316,
    "knee_angle_r": -0.393922,
    "ankle_angle_r": 0.104714,

    "hip_flexion_l": 0.198813,
    "knee_angle_l": -1.03755,
    "ankle_angle_l": -0.348473,

    "lumbar_extension": 0.0,
}


GAIT_INITIATION_INITIAL_VELOCITY = {
    "pelvis_tilt": -0.0895989,
    "pelvis_tx": 1.0757,
    "pelvis_ty": 0.1543,

    "hip_flexion_r": -1.35971,
    "knee_angle_r": 0.267883,
    "ankle_angle_r": 0.840122,

    "hip_flexion_l": 3.34368,
    "knee_angle_l": -3.15281,
    "ankle_angle_l": 1.26642,
}


def set_gait_initiation_pose(
    model,
    data,
    qpos_by_name=None,
    qvel_by_name=None,
    do_forward=True,
):
    """
    歩行開始動作用の初期姿勢と初期速度を設定する。
    """

    if qpos_by_name is None:
        qpos_by_name = GAIT_INITIATION_INITIAL_POSE

    for joint_name, value in qpos_by_name.items():
        joint_id = model.joint(joint_name).id
        qpos_adr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_adr] = value

    data.qvel[:] = 0.0
    data.qacc[:] = 0.0

    if qvel_by_name is None:
        qvel_by_name = GAIT_INITIATION_INITIAL_VELOCITY

    for joint_name, value in qvel_by_name.items():
        joint_id = model.joint(joint_name).id
        qvel_adr = model.jnt_dofadr[joint_id]
        data.qvel[qvel_adr] = value

    if do_forward:
        mj_forward(model, data)