# simulation/objectives/wang2012.py

import numpy as np


def Wang2012(
    u,
    a,
    l_MTU,
    v_MTU,
    F_MTU,
    mass,
    l_mtu_opt,
    slow_twitch_ratio,
):
    u = np.asarray(u)
    a = np.asarray(a)
    l_MTU = np.asarray(l_MTU)
    v_MTU = np.asarray(v_MTU)
    F_MTU = np.asarray(F_MTU)

    mass = np.asarray(mass)
    l_mtu_opt = np.asarray(l_mtu_opt)
    s = np.asarray(slow_twitch_ratio)

    l_ce_norm = l_MTU / l_mtu_opt
    v_CE = v_MTU

    # MuJoCoのactuator_forceは符号が扱いにくいので力の大きさとして扱う
    F_CE = np.abs(F_MTU)

    fa = (
        40.0 * s * np.sin(np.pi / 2.0 * u)
        + 133.0 * (1.0 - s) * (1.0 - np.cos(np.pi / 2.0 * u))
    )

    fm = (
        74.0 * s * np.sin(np.pi / 2.0 * a)
        + 111.0 * (1.0 - s) * (1.0 - np.cos(np.pi / 2.0 * a))
    )

    g = np.where(
        l_ce_norm < 0.5,
        0.5,
        np.where(
            l_ce_norm < 1.0,
            l_ce_norm,
            np.where(
                l_ce_norm < 1.5,
                -2.0 * l_ce_norm + 3.0,
                0.0,
            ),
        ),
    )

    effort_a = mass * fa
    effort_m = mass * g * fm

    # 短縮熱：v_CE < 0 を短縮とみなす
    effort_s = 0.25 * F_CE * np.maximum(0.0, -v_CE)

    # 正の機械仕事
    effort_w = np.maximum(0.0, F_CE * v_CE)

    muscle_energy = effort_a + effort_m + effort_s + effort_w
    total_energy = np.sum(muscle_energy)

    # print("Ea =", effort_a)
    # print("Em =", effort_m)
    # print("Es =", effort_s)
    # print("Ew =", effort_w)
    # print("muscle_energy =", muscle_energy)
    # print("total_energy =", total_energy)

    return total_energy, muscle_energy