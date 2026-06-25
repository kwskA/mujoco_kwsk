import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path( r"model/1422model.xml")
data = mujoco.MjData(model)

# キーフレーム初期化
data.qpos[:] = model.key_qpos[0]
data.qvel[:] = 0
data.qacc[:] = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
