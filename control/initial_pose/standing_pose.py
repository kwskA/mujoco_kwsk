# control/initial_pose/standing_pose.py

from mujoco import mj_forward


STANDING_INITIAL_POSE = {
    "pelvis_tx": 0.0,
    "pelvis_ty": 0.950000,
    "pelvis_tilt": 0.0,

    "hip_flexion_r": 0.0,
    "knee_angle_r": 0.0,
    "ankle_angle_r": 0.0,

    "hip_flexion_l": 0.0,
    "knee_angle_l": 0.0,
    "ankle_angle_l": 0.0,

    "lumbar_extension": 0.0,
}


def set_standing_pose(
    model,
    data,
    qpos_by_name=None,
    qvel_by_name=None,
    do_forward=True,
):
    """
    立位動作用の初期姿勢と初期速度を設定する。
    """

    if qpos_by_name is None:
        qpos_by_name = STANDING_INITIAL_POSE

    for joint_name, value in qpos_by_name.items():
        joint_id = model.joint(joint_name).id
        qpos_adr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_adr] = value

    data.qvel[:] = 0.0
    data.qacc[:] = 0.0

    if qvel_by_name is not None:
        for joint_name, value in qvel_by_name.items():
            joint_id = model.joint(joint_name).id
            qvel_adr = model.jnt_dofadr[joint_id]
            data.qvel[qvel_adr] = value

    if do_forward:
        mj_forward(model, data)