# test_control.py

import numpy as np
import mujoco

from mujoco import MjModel, MjData, mj_forward

from control.control_methods.p_controller import PController
from control.control_methods.pd_controller import PDController
from control.control_methods.extended_p_controller import ExtendedPController

from control.phases.standing_controller import StandingController
from control.phases.gait_initiation_controller import GaitInitiationController
from control.phases.sequential_controller import SequentialController


def main():
    model_path = "model/1018model.xml"

    muscles = [
        "hamstrings_r", "bifemsh_r", "glut_max_r",
        "iliopsoas_r", "rect_fem_r", "vasti_r",
        "gastroc_r", "soleus_r", "tib_ant_r",

        "hamstrings_l", "bifemsh_l", "glut_max_l",
        "iliopsoas_l", "rect_fem_l", "vasti_l",
        "gastroc_l", "soleus_l", "tib_ant_l",
    ]

    model = MjModel.from_xml_path(model_path)
    data = MjData(model)

    data.qpos[:] = model.key_qpos[0].copy()
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mj_forward(model, data)

    tendon_ids = np.array([
        model.tendon(f"{m}_tendon").id
        for m in muscles
    ], dtype=int)

    init_len = np.array([
        data.ten_length[t_id]
        for t_id in tendon_ids
    ])

    print("model.nu =", model.nu)
    print("num_muscles =", len(muscles))
    print("init_len shape =", init_len.shape)

    # -------------------------
    # PController test
    # -------------------------
    p_method = PController(model, muscles)
    p_params = p_method.make_initial_params(
        init_Kp=10.0,
        init_target_length=init_len,
    )
    p_method.set_params_from_vector(p_params)
    ctrl_p = p_method.compute_ctrl(data)

    print("\n[PController]")
    print("param_dim =", p_method.get_expected_param_dim())
    print("ctrl shape =", ctrl_p.shape)
    print("ctrl min/max =", ctrl_p.min(), ctrl_p.max())

    # -------------------------
    # PDController test
    # -------------------------
    pd_method = PDController(model, muscles)
    pd_params = pd_method.make_initial_params(
        init_Kp=10.0,
        init_Kd=2.0,
        init_target_length=init_len,
    )
    pd_method.set_params_from_vector(pd_params)
    ctrl_pd = pd_method.compute_ctrl(data)

    print("\n[PDController]")
    print("param_dim =", pd_method.get_expected_param_dim())
    print("ctrl shape =", ctrl_pd.shape)
    print("ctrl min/max =", ctrl_pd.min(), ctrl_pd.max())

    # -------------------------
    # ExtendedPController test
    # -------------------------
    ext_method = ExtendedPController(model, muscles)
    ext_params = ext_method.make_initial_params(
        init_Kp=10.0,
        init_target_length=init_len,
    )
    ext_method.set_params_from_vector(ext_params)
    ctrl_ext = ext_method.compute_ctrl(data)

    print("\n[ExtendedPController]")
    print("param_dim =", ext_method.get_expected_param_dim())
    print("ctrl shape =", ctrl_ext.shape)
    print("ctrl min/max =", ctrl_ext.min(), ctrl_ext.max())

    # -------------------------
    # Phase controller test
    # -------------------------
    standing = StandingController(
        name="Standing",
        control_method=PDController(model, muscles),
    )

    gait_init1 = GaitInitiationController(
        name="GaitInitiation1",
        control_method=PDController(model, muscles),
    )

    seq = SequentialController(
        phases=[standing, gait_init1],
        durations=[1.0, None],
    )

    x0 = seq.make_initial_params(
        init_Kp=10.0,
        init_Kd=2.0,
        init_target_length=init_len,
    )

    seq.set_params_from_vector(x0)

    ctrl_seq = seq.compute_ctrl(data)

    print("\n[SequentialController]")
    print("param_dim =", seq.get_expected_param_dim())
    print("x0 shape =", x0.shape)
    print("ctrl shape =", ctrl_seq.shape)
    print("ctrl min/max =", ctrl_seq.min(), ctrl_seq.max())

    print("\n✅ controlフォルダの基本テスト完了")


if __name__ == "__main__":
    main()