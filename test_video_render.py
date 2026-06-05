# test_video_renderer.py

import numpy as np
from pathlib import Path

from mujoco import MjModel, MjData, mj_forward

from control.control_methods.pd_controller import PDController
from control.phases.standing_controller import StandingController
from control.phases.gait_initiation_controller import GaitInitiationController
from control.phases.sequential_controller import SequentialController

from render.video_renderer import VideoRenderer


ROOT_DIR = Path(__file__).parent
MODEL_PATH = str(ROOT_DIR / "model" / "1018model.xml")

MUSCLES = [
    "hamstrings_r", "bifemsh_r", "glut_max_r",
    "iliopsoas_r", "rect_fem_r", "vasti_r",
    "gastroc_r", "soleus_r", "tib_ant_r",

    "hamstrings_l", "bifemsh_l", "glut_max_l",
    "iliopsoas_l", "rect_fem_l", "vasti_l",
    "gastroc_l", "soleus_l", "tib_ant_l",
]


def main():
    model = MjModel.from_xml_path(MODEL_PATH)
    data = MjData(model)

    data.qpos[:] = model.key_qpos[0].copy()
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mj_forward(model, data)

    initial_qpos = data.qpos.copy()

    tendon_ids = np.array([
        model.tendon(f"{m}_tendon").id
        for m in MUSCLES
    ], dtype=int)

    actuator_ids = np.array([
        model.actuator(m).id
        for m in MUSCLES
    ], dtype=int)

    init_len = np.array([
        data.ten_length[t_id]
        for t_id in tendon_ids
    ])

    standing = StandingController(
        name="Standing",
        control_method=PDController(model, MUSCLES),
    )

    gait_init = GaitInitiationController(
        name="GaitInitiation1",
        control_method=PDController(model, MUSCLES),
    )

    controller = SequentialController(
        phases=[standing, gait_init],
        durations=[0.5, None],
    )

    x0 = controller.make_initial_params(
        init_Kp=10.0,
        init_Kd=2.0,
        init_target_length=init_len,
    )

    controller.set_params_from_vector(x0)

    save_dir = ROOT_DIR / "test_results" / "video"
    save_dir.mkdir(parents=True, exist_ok=True)

    video_path = save_dir / "test_video.mp4"

    renderer = VideoRenderer(
        model=model,
        controller=controller,
        sim_steps=200,
        initial_qpos=initial_qpos,
        width=400,
        height=400,
        fps=200,
    )

    renderer.render(
        save_path=str(video_path),
        tendon_ids=tendon_ids,
        actuator_ids=actuator_ids,
        color_tendons=True,
    )

    print("\n[VideoRenderer test]")
    print("video_path =", video_path)
    print("\n✅ 動画出力テスト完了")


if __name__ == "__main__":
    main()