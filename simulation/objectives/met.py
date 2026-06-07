import numpy as np
from objectives.muscle_data import muscle_mass, length_opt, f_max, v_max

def muscle_metabolic_energy_vectorized(m, A, length_opt_arr, V_CE, F_CE, r=0.5):
    """
    全て numpy 配列で計算する高速版
    """
    m = np.asarray(m)
    A = np.asarray(A)
    li = np.asarray(length_opt_arr)
    vi = np.asarray(V_CE)
    fi = np.asarray(F_CE)

    # avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        v_norm = np.where(li != 0, vi / li, 0.0)

    AMdot = 89.0 * m * (A ** 0.6)

    # Sdot: piecewise depending on v_norm >= 0 (shortening) or < 0 (lengthening)
    pos_mask = v_norm >= 0
    # when shortening (v_norm >= 0)
    alphaS_fast = 15.3
    alphaS_slow = 25.0
    alphaL = 100.0

    Sdot = np.empty_like(v_norm)
    Sdot[pos_mask] = np.abs(m[pos_mask] * ( -((alphaS_slow * v_norm[pos_mask] * r) +
                                              (alphaS_fast * v_norm[pos_mask] * (1 - r))) * A[pos_mask]**2))
    Sdot[~pos_mask] = np.abs(m[~pos_mask] * (alphaL * (-v_norm[~pos_mask]) * A[~pos_mask]))

    Wdot = np.abs(fi * vi)

    Edot = np.sum(AMdot + Sdot + Wdot)
    Edot = np.sum(AMdot)
    return Edot

# def muscle_metabolic_energy_wang2012(m, A, l_opt, V_CE, F_CE, F_max, fiber_type_ratio=0.5):
#     """
#     Wang2012型の簡易代謝モデル（ベクトル化）
    
#     m: 筋質量 [kg]
#     A: 筋活性 [0-1]
#     l_opt: 最適筋長 [m]
#     V_CE: 収縮速度 [m/s]
#     F_CE: 筋力 [N]
#     F_max: 最大筋力 [N]
#     fiber_type_ratio: fast-twitch比率（0〜1）
#     """

#     m = np.asarray(m)
#     A = np.asarray(A)
#     l_opt = np.asarray(l_opt)
#     v = np.asarray(V_CE)
#     f = np.asarray(F_CE)
#     f_max = np.asarray(F_max)

#     # 正規化速度
#     with np.errstate(divide='ignore', invalid='ignore'):
#         v_norm = np.where(l_opt != 0, v / l_opt, 0.0)

#     # ==== 1. Activation heat ====
#     # Wang2012: 活性依存
#     A_dot = 40.0 * m * A

#     # ==== 2. Maintenance heat ====
#     # 張力と活性に依存
#     M_dot = 0.5 * m * A * (f / f_max)

#     # ==== 3. Shortening / Lengthening heat ====
#     S_dot = np.zeros_like(v)

#     shortening = v_norm > 0
#     lengthening = v_norm < 0

#     # 短縮性収縮
#     S_dot[shortening] = (
#         m[shortening]
#         * A[shortening]
#         * (v_norm[shortening] ** 2)
#         * (15.0 * fiber_type_ratio + 5.0 * (1 - fiber_type_ratio))
#     )

#     # 伸張性収縮（消費は小さい）
#     S_dot[lengthening] = (
#         m[lengthening]
#         * A[lengthening]
#         * (np.abs(v_norm[lengthening]) ** 1.5)
#         * 2.0
#     )

#     # ==== 4. Mechanical work ====
#     W_dot = np.maximum(0.0, f * v)

#     # ==== Total ====
#     E_dot = A_dot + M_dot + S_dot + W_dot

#     return np.sum(E_dot)


def muscle_metabolic_energy_wang2012(a, v, f):
    """
    SCONE-style Wang2012 metabolic energy for one timestep.

    Parameters
    ----------
    a : np.ndarray, shape (n_muscles,)
        Muscle activations [0,1]
    f_norm : np.ndarray, shape (n_muscles,)
        Normalized muscle force
    v_norm : np.ndarray, shape (n_muscles,)
        Normalized muscle fiber velocity
        (+) shortening, (-) lengthening
    f_max : np.ndarray, shape (n_muscles,)
        Max isometric force [N]
    Returns
    -------
    energy : float
        Metabolic energy for this timestep [J]
    """

    A_AM = 1.28
    A_SH = 0.25
    A_LEN = 0.0
    EFFICIENCY = 0.25
    v *= -1
    f *= -1
    v_norm = v / v_max
    f_norm = f / f_max

    # --- Activation + maintenance heat ---
    heat_am = A_AM * a * f_max  # [W]

    # --- Shortening / lengthening heat ---
    heat_sl = np.zeros_like(a)
    shortening = v_norm > 0.0

    heat_sl[shortening] = A_SH * a[shortening] * v_norm[shortening] * f_max[shortening]
    heat_sl[~shortening] = A_LEN * a[~shortening] * (-v_norm[~shortening]) * f_max[~shortening]

    # --- Mechanical work term (positive work only) ---
    power = f * v_norm  # [W]
    mech = np.where(power > 0.0, power / EFFICIENCY, 0.0)

    # --- Total metabolic rate [W] ---
    metabolic_rate = heat_am + heat_sl + mech

    return np.sum(metabolic_rate)

import numpy as np

def Wang2012(u, a, l_MTU, v_MTU, F_MTU, mass, l_mtu_opt, slow_twitch_ratio):

    """
    Wang2012 metabolic energy model
    (vectorized implementation for MuJoCo)

    Parameters
    ----------
    u : ndarray
        excitation (= data.ctrl)

    a : ndarray
        activation (= data.act)

    l_MTU : ndarray
        MTU length (= data.actuator_length) [m]

    v_MTU : ndarray
        MTU velocity (= data.actuator_velocity) [m/s]
        shortening direction -> negative

    F_MTU : ndarray
        MTU force (= data.actuator_force) [N]

    Returns
    -------
    total_energy : float
        total metabolic power [W]

    muscle_energy : ndarray
        metabolic power of each muscle [W]
    """

    u = np.asarray(u)
    a = np.asarray(a)

    l_MTU = np.asarray(l_MTU)
    v_MTU = np.asarray(v_MTU)

    # MuJoCo actuator_force becomes negative during shortening,
    # therefore absolute value is used.
    F_MTU = np.asarray(F_MTU)

    # ==========================================================
    # Approximation of normalized CE length
    #
    # OpenSim-like CE reconstruction:
    #   l_MTU = l_T + l_CE*cos(alpha)
    #
    # is NOT used here because:
    #
    # 1. MuJoCo does not explicitly expose CE/SE states
    # 2. tendon dynamics are not solved explicitly
    # 3. tendon_slack_length may exceed minimum actuator length
    # 4. direct reconstruction can produce non-physical l_CE < 0
    #
    # Therefore:
    #
    #   normalized l_CE ≈ normalized l_MTU
    #
    # is used as a robust approximation:
    #
    #   l_ce_norm ≈ l_MTU / l_MTU_opt
    #
    # This approximation is commonly used in fast
    # musculoskeletal optimization / RL settings.
    # ==========================================================

    l_ce_norm = l_MTU / l_mtu_opt

    # ==========================================================
    # Approximation of CE velocity
    #
    # Ideally:
    #
    #   v_CE = (v_MTU - v_T) / cos(alpha)
    #
    # However MuJoCo does not explicitly solve tendon dynamics.
    #
    # Therefore rigid tendon approximation is used:
    #
    #   v_CE ≈ v_MTU
    #
    # This keeps consistency with:
    #
    #   l_ce_norm ≈ l_MTU / l_MTU_opt
    # ==========================================================

    v_CE = v_MTU

    # ==========================================================
    # Approximation of active CE force
    #
    # MuJoCo actuator_force already contains muscle active force
    # generated from FLV-like muscle dynamics.
    #
    # Therefore:
    #
    #   F_CE ≈ F_MTU
    #
    # is used directly.
    #
    # Explicit passive elastic force separation is NOT performed
    # to avoid double-counting passive force components.
    # ==========================================================

    F_CE = F_MTU

    # ==========================================================
    # slow twitch fiber ratio
    # ==========================================================

    s = slow_twitch_ratio

    # ==========================================================
    # activation heat rate
    # ==========================================================

    fa = 40.0 * s * np.sin(np.pi / 2.0 * u) + 133.0 * (1.0 - s) * (1.0 - np.cos(np.pi / 2.0 * u))

    # ==========================================================
    # maintenance heat rate
    # ==========================================================

    fm = 74.0 * s * np.sin(np.pi / 2.0 * a) + 111.0 * (1.0 - s) * (1.0 - np.cos(np.pi / 2.0 * a))

    # ==========================================================
    # length dependency term
    # ==========================================================

    g = np.where(
        l_ce_norm < 0.5,
        0.5,
        np.where(
            l_ce_norm < 1.0,
            l_ce_norm,
            np.where(
                l_ce_norm < 1.5,
                -2.0 * l_ce_norm + 3.0,
                0.0
            )
        )
    )

    # ==========================================================
    # activation energy
    # ==========================================================

    effort_a = mass * fa

    # ==========================================================
    # maintenance energy
    # ==========================================================

    effort_m = mass * g * fm

    # ==========================================================
    # shortening heat rate
    #
    # shortening -> v_CE < 0
    # ==========================================================

    effort_s = np.maximum(0.0, 0.25 * F_CE * v_CE)

    # ==========================================================
    # mechanical work rate
    # ==========================================================

    effort_w = np.maximum(0.0, F_CE * v_CE)

    # ==========================================================
    # total muscle metabolic rate
    # ==========================================================

    muscle_energy = effort_a + effort_m + effort_s + effort_w

    total_energy = np.sum(muscle_energy)

    return total_energy

