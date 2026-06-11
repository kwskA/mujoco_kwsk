from dataclasses import dataclass
import numpy as np


# ==========================================================
# Muscle parameter class
# ==========================================================

@dataclass
class MuscleParameters:

    # maximum isometric force [N]
    # cf. gait2354_simbody.osim
    f_max: float

    # optimal fiber length [m]
    # cf. gait2354_simbody.osim
    l_opt: float

    # maximum contraction velocity [l_opt / s]
    # cf. gait2354_simbody.osim
    v_max: float

    # slow twitch fiber ratio [0-1]
    # cf. https://github.com/opensim-org/opensim-models/blob/master/Tutorials/Design_to_Reduce_Metabolic_Cost/Scripts/metabolicsSlowTwitchRatios_Gait2392.txt
    slow_twitch_ratio: float

    # muscle_mass [kg]
    # muscle_mass [kg] = (f_max / specific_tension) * density * length_opt
    # specific_tension = 0.25e6 [N/m^2]
    # density = 1059.7 [kg/m^3]
    # https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1Umberger2010MuscleMetabolicsProbe__MetabolicMuscleParameter.html
    muscle_mass: float

    # tendon slack length [m]
    # cf. gait2354_simbody.osim
    tendon_slack_length: float

    # pennation angle at optimal fiber length [rad]
    # cf. gait2354_simbody.osim
    pennation_angle: float

    # optimal MTU length [m]
    #
    # used for:
    # l_ce_norm ≈ l_MTU / l_MTU_opt
    #
    # l_mtu_opt = tendon_slack_length + l_opt*cos(pnnation_angle)
    l_mtu_opt: float

# ==========================================================
# gait2354 muscle parameters
#
# IMPORTANT:
# right and left muscles are intentionally stored separately
# because future asymmetric parameter tuning may be required.
# ==========================================================

muscle_data = {

    # ======================================================
    # glut_med1
    # ======================================================
    
    "glut_med1_r": MuscleParameters(
        f_max=1119,
        l_opt=0.0535,
        v_max=10.0,
        slow_twitch_ratio=0.55,
        muscle_mass=0.253762,
        tendon_slack_length=0.078,
        pennation_angle=0.13962634,
        l_mtu_opt=0.130979,
    ),

    "glut_med1_l": MuscleParameters(
        f_max=1119,
        l_opt=0.0535,
        v_max=10.0,
        slow_twitch_ratio=0.55,
        muscle_mass=0.253762,
        tendon_slack_length=0.078,
        pennation_angle=0.13962634,
        l_mtu_opt=0.130979,
    ),

    # ======================================================
    # glut_med2
    # ======================================================
    
    "glut_med2_r": MuscleParameters(
        f_max=873,
        l_opt=0.0845,
        v_max=10.0,
        slow_twitch_ratio=0.55,
        muscle_mass=0.312690,
        tendon_slack_length=0.053,
        pennation_angle=0,
        l_mtu_opt=0.137500,
    ),

    "glut_med2_l": MuscleParameters(
        f_max=873,
        l_opt=0.0845,
        v_max=10.0,
        slow_twitch_ratio=0.55,
        muscle_mass=0.312690,
        tendon_slack_length=0.053,
        pennation_angle=0,
        l_mtu_opt=0.137500,
    ),
    
    # ======================================================
    # glut_med3
    # ======================================================
    
    "glut_med3_r": MuscleParameters(
        f_max=1000,
        l_opt=0.0646,
        v_max=10.0,
        slow_twitch_ratio=0.55,
        muscle_mass=0.273826,
        tendon_slack_length=0.053,
        pennation_angle=0.33161256,
        l_mtu_opt=0.114080,
    ),

    "glut_med3_l": MuscleParameters(
        f_max=1000,
        l_opt=0.0646,
        v_max=10.0,
        slow_twitch_ratio=0.55,
        muscle_mass=0.273826,
        tendon_slack_length=0.053,
        pennation_angle=0.33161256,
        l_mtu_opt=0.114080,
    ),

    # ======================================================
    # bifemlh
    # ======================================================
    
    "bifemlh_r": MuscleParameters(
        f_max=2700,
        l_opt=0.109,
        v_max=10.0,
        slow_twitch_ratio=0.5425,
        muscle_mass=1.247479,
        tendon_slack_length=0.326,
        pennation_angle=0,
        l_mtu_opt=0.435000,
    ),

    "bifemlh_l": MuscleParameters(
        f_max=2700,
        l_opt=0.109,
        v_max=10.0,
        slow_twitch_ratio=0.5425,
        muscle_mass=1.24747,
        tendon_slack_length=0.326,
        pennation_angle=0,
        l_mtu_opt=0.435000,
    ),

    # ======================================================
    # bifemsh
    # ======================================================
    
    "bifemsh_r": MuscleParameters(
        f_max=804,
        l_opt=0.173,
        v_max=10.0,
        slow_twitch_ratio=0.529,
        muscle_mass=0.589583,
        tendon_slack_length=0.089,
        pennation_angle=0.40142573,
        l_mtu_opt=0.248247,
    ),

    "bifemsh_l": MuscleParameters(
        f_max=804,
        l_opt=0.173,
        v_max=10.0,
        slow_twitch_ratio=0.529,
        muscle_mass=0.589583,
        tendon_slack_length=0.089,
        pennation_angle=0.40142573,
        l_mtu_opt=0.248247,
    ),

    # ======================================================
    # sar
    # ======================================================
    
    "sar_r": MuscleParameters(
        f_max=156,
        l_opt=0.52,
        v_max=10.0,
        slow_twitch_ratio=0.5,
        muscle_mass=0.343851,
        tendon_slack_length=0.1,
        pennation_angle=0,
        l_mtu_opt=0.620000,
    ),

    "sar_l": MuscleParameters(
        f_max=156,
        l_opt=0.52,
        v_max=10.0,
        slow_twitch_ratio=0.5,
        muscle_mass=0.343851,
        tendon_slack_length=0.1,
        pennation_angle=0,
        l_mtu_opt=0.620000,
    ),

    # ======================================================
    # add_mag2
    # ======================================================
    
    "add_mag2_r": MuscleParameters(
        f_max=2343,
        l_opt=0.121,
        v_max=10.0,
        slow_twitch_ratio=0.552,
        muscle_mass=1.201713,
        tendon_slack_length=0.12,
        pennation_angle=0.05235988,
        l_mtu_opt=0.240834,
    ),

    "add_mag2_l": MuscleParameters(
        f_max=2343,
        l_opt=0.121,
        v_max=10.0,
        slow_twitch_ratio=0.552,
        muscle_mass=1.201713,
        tendon_slack_length=0.12,
        pennation_angle=0.05235988,
        l_mtu_opt=0.240834,
    ),

    # ======================================================
    # tfl
    # ======================================================
    
    "tfl_r": MuscleParameters(
        f_max=233,
        l_opt=0.095,
        v_max=10.0,
        slow_twitch_ratio=0.5,
        muscle_mass=0.093826,
        tendon_slack_length=0.425,
        pennation_angle=0.05235988,
        l_mtu_opt=0.519870,
    ),

    "tfl_l": MuscleParameters(
        f_max=233,
        l_opt=0.095,
        v_max=10.0,
        slow_twitch_ratio=0.5,
        muscle_mass=0.093826,
        tendon_slack_length=0.425,
        pennation_angle=0.05235988,
        l_mtu_opt=0.519870,
    ),

    # ======================================================
    # pect
    # ======================================================
    
    "pect_r": MuscleParameters(
        f_max=266,
        l_opt=0.1,
        v_max=10.0,
        slow_twitch_ratio=0.5,
        muscle_mass=0.112752,
        tendon_slack_length=0.033,
        pennation_angle=0,
        l_mtu_opt=0.133000,
    ),

    "pect_l": MuscleParameters(
        f_max=266,
        l_opt=0.1,
        v_max=10.0,
        slow_twitch_ratio=0.5,
        muscle_mass=0.112752,
        tendon_slack_length=0.033,
        pennation_angle=0,
        l_mtu_opt=0.133000,
    ),

    # ======================================================
    # grac
    # ======================================================
    
    "grac_r": MuscleParameters(
        f_max=162,
        l_opt=0.352,
        v_max=10.0,
        slow_twitch_ratio=0.5,
        muscle_mass=0.241713,
        tendon_slack_length=0.126,
        pennation_angle=0.05235988,
        l_mtu_opt=0.477518,
    ),

    "grac_l": MuscleParameters(
        f_max=162,
        l_opt=0.352,
        v_max=10.0,
        slow_twitch_ratio=0.5,
        muscle_mass=0.241713,
        tendon_slack_length=0.126,
        pennation_angle=0.05235988,
        l_mtu_opt=0.477518,
    ),

    # ======================================================
    # glut_max1
    # ======================================================
    
    "glut_max1_r": MuscleParameters(
        f_max=573,
        l_opt=0.142,
        v_max=10.0,
        slow_twitch_ratio=0.55,
        muscle_mass=0.344894,
        tendon_slack_length=0.125,
        pennation_angle=0.08726646,
        l_mtu_opt=0.266460,
    ),

    "glut_max1_l": MuscleParameters(
        f_max=573,
        l_opt=0.142,
        v_max=10.0,
        slow_twitch_ratio=0.55,
        muscle_mass=0.344894,
        tendon_slack_length=0.1250,
        pennation_angle=0.08726646,
        l_mtu_opt=0.266460,
    ),

    # ======================================================
    # glut_max2
    # ======================================================
    
    "glut_max2_r": MuscleParameters(
        f_max=819,
        l_opt=0.147,
        v_max=10.0,
        slow_twitch_ratio=0.55,
        muscle_mass=0.510322,
        tendon_slack_length=0.127,
        pennation_angle=0,
        l_mtu_opt=0.274000,
    ),

    "glut_max2_l": MuscleParameters(
        f_max=819,
        l_opt=0.147,
        v_max=10.0,
        slow_twitch_ratio=0.55,
        muscle_mass=0.510322,
        tendon_slack_length=0.127,
        pennation_angle=0,
        l_mtu_opt=0.274000,
    ),

    # ======================================================
    # glut_max3
    # ======================================================
    
    "glut_max3_r": MuscleParameters(
        f_max=552,
        l_opt=0.144,
        v_max=10.0,
        slow_twitch_ratio=0.55,
        muscle_mass=0.336934,
        tendon_slack_length=0.145,
        pennation_angle=0.08726646,
        l_mtu_opt=0.288452,
    ),

    "glut_max3_l": MuscleParameters(
        f_max=552,
        l_opt=0.144,
        v_max=10.0,
        slow_twitch_ratio=0.55,
        muscle_mass=0.336934,
        tendon_slack_length=0.145,
        pennation_angle=0.08726646,
        l_mtu_opt=0.288452,
    ),

    # ======================================================
    # iliacus
    # ======================================================
    
    "iliacus_r": MuscleParameters(
        f_max=1073,
        l_opt=0.1,
        v_max=10.0,
        slow_twitch_ratio=0.5,
        muscle_mass=0.454823,
        tendon_slack_length=0.1,
        pennation_angle=0.12217305,
        l_mtu_opt=0.199255,
    ),

    "iliacus_l": MuscleParameters(
        f_max=1073,
        l_opt=0.1,
        v_max=10.0,
        slow_twitch_ratio=0.5,
        muscle_mass=0.454823,
        tendon_slack_length=0.1,
        pennation_angle=0.12217305,
        l_mtu_opt=0.199255,
    ),

    # ======================================================
    # psoas
    # ======================================================
    
    "psoas_r": MuscleParameters(
        f_max=1113,
        l_opt=0.1,
        v_max=10.0,
        slow_twitch_ratio=0.5,
        muscle_mass=0.471778,
        tendon_slack_length=0.16,
        pennation_angle=0.13962634,
        l_mtu_opt=0.259027,
    ),

    "psoas_l": MuscleParameters(
        f_max=1113,
        l_opt=0.1,
        v_max=10.0,
        slow_twitch_ratio=0.5,
        muscle_mass=0.471778,
        tendon_slack_length=0.16,
        pennation_angle=0.13962634,
        l_mtu_opt=0.259027,
    ),

    # ======================================================
    # quad_fem
    # ======================================================
    
    "quad_fem_r": MuscleParameters(
        f_max=381,
        l_opt=0.054,
        v_max=10.0,
        slow_twitch_ratio=0.5,
        muscle_mass=0.087209,
        tendon_slack_length=0.024,
        pennation_angle=0,
        l_mtu_opt=0.078000,
    ),

    "quad_fem_l": MuscleParameters(
        f_max=381,
        l_opt=0.054,
        v_max=10.0,
        slow_twitch_ratio=0.5,
        muscle_mass=0.087209,
        tendon_slack_length=0.024,
        pennation_angle=0,
        l_mtu_opt=0.078000,
    ),

    # ======================================================
    # gem
    # ======================================================
    
    "gem_r": MuscleParameters(
        f_max=164,
        l_opt=0.024,
        v_max=10.0,
        slow_twitch_ratio=0.5,
        muscle_mass=0.016684,
        tendon_slack_length=0.039,
        pennation_angle=0,
        l_mtu_opt=0.063000,
    ),

    "gem_l": MuscleParameters(
        f_max=164,
        l_opt=0.024,
        v_max=10.0,
        slow_twitch_ratio=0.5,
        muscle_mass=0.016684,
        tendon_slack_length=0.039,
        pennation_angle=0,
        l_mtu_opt=0.063000,
    ),

    # ======================================================
    # peri
    # ======================================================
    
    "peri_r": MuscleParameters(
        f_max=444,
        l_opt=0.026,
        v_max=10.0,
        slow_twitch_ratio=0.5,
        muscle_mass=0.048933,
        tendon_slack_length=0.115,
        pennation_angle=0.17453293,
        l_mtu_opt=0.140605,
    ),

    "peri_l": MuscleParameters(
        f_max=444,
        l_opt=0.026,
        v_max=10.0,
        slow_twitch_ratio=0.5,
        muscle_mass=0.048933,
        tendon_slack_length=0.115,
        pennation_angle=0.17453293,
        l_mtu_opt=0.140605,
    ),


    # ======================================================
    # rect_fem
    # ======================================================
    
    "rect_fem_r": MuscleParameters(
        f_max=1169,
        l_opt=0.114,
        v_max=10.0,
        slow_twitch_ratio=0.3865,
        muscle_mass=0.564888,
        tendon_slack_length=0.31,
        pennation_angle=0.08726646,
        l_mtu_opt=0.423566,
    ),

    "rect_fem_l": MuscleParameters(
        f_max=1169,
        l_opt=0.114,
        v_max=10.0,
        slow_twitch_ratio=0.3865,
        muscle_mass=0.564888,
        tendon_slack_length=0.31,
        pennation_angle=0.08726646,
        l_mtu_opt=0.423566,
    ),

    # ======================================================
    # vas_int
    # ======================================================
    
    "vas_int_r": MuscleParameters(
        f_max=5000,
        l_opt=0.107,
        v_max=10.0,
        slow_twitch_ratio=0.543,
        muscle_mass=2.267758,
        tendon_slack_length=0.116,
        pennation_angle=0.05235988,
        l_mtu_opt=0.222853,
    ),

    "vas_int_l": MuscleParameters(
        f_max=5000,
        l_opt=0.107,
        v_max=10.0,
        slow_twitch_ratio=0.543,
        muscle_mass=2.267758,
        tendon_slack_length=0.116,
        pennation_angle=0.05235988,
        l_mtu_opt=0.222853,
    ),

    # ======================================================
    # med_gas
    # ======================================================
    
    "med_gas_r": MuscleParameters(
        f_max=2500,
        l_opt=0.09,
        v_max=10.0,
        slow_twitch_ratio=0.566,
        muscle_mass=0.953730,
        tendon_slack_length=0.36,
        pennation_angle=0.29670597,
        l_mtu_opt=0.446067,
    ),

    "med_gas_l": MuscleParameters(
        f_max=2500,
        l_opt=0.09,
        v_max=10.0,
        slow_twitch_ratio=0.566,
        muscle_mass=0.953730,
        tendon_slack_length=0.36,
        pennation_angle=0.29670597,
        l_mtu_opt=0.446067,
    ),

    # ======================================================
    # soleus
    # ======================================================
    
    "soleus_r": MuscleParameters(
        f_max=4000,
        l_opt=0.05,
        v_max=10.0,
        slow_twitch_ratio=0.803,
        muscle_mass=0.847760,
        tendon_slack_length=0.25,
        pennation_angle=0.43633231,
        l_mtu_opt=0.295315,
    ),

    "soleus_l": MuscleParameters(
        f_max=4000,
        l_opt=0.05,
        v_max=10.0,
        slow_twitch_ratio=0.803,
        muscle_mass=0.847760,
        tendon_slack_length=0.25,
        pennation_angle=0.43633231,
        l_mtu_opt=0.295315,
    ),

    # ======================================================
    # tib_post
    # ======================================================
    
    "tib_post_r": MuscleParameters(
        f_max=3600,
        l_opt=0.031,
        v_max=10.0,
        slow_twitch_ratio=0.6,
        muscle_mass=0.473050,
        tendon_slack_length=0.31,
        pennation_angle=0.20943951,
        l_mtu_opt=0.340323,
    ),

    "tib_post_l": MuscleParameters(
        f_max=3600,
        l_opt=0.031,
        v_max=10.0,
        slow_twitch_ratio=0.6,
        muscle_mass=0.473050,
        tendon_slack_length=0.31,
        pennation_angle=0.20943951,
        l_mtu_opt=0.340323,
    ),

    # ======================================================
    # tib_ant
    # ======================================================
    
    "tib_ant_r": MuscleParameters(
        f_max=3000,
        l_opt=0.098,
        v_max=10.0,
        slow_twitch_ratio=0.7,
        muscle_mass=1.246207,
        tendon_slack_length=0.223,
        pennation_angle=0.08726646,
        l_mtu_opt=0.320627,
    ),

    "tib_ant_l": MuscleParameters(
        f_max=3000,
        l_opt=0.098,
        v_max=10.0,
        slow_twitch_ratio=0.7,
        muscle_mass=1.246207,
        tendon_slack_length=0.223,
        pennation_angle=0.08726646,
        l_mtu_opt=0.320627,
    ),

    # ======================================================
    # ercspn
    # ======================================================
    
    "ercspn_r": MuscleParameters(
        f_max=2500,
        l_opt=0.12,
        v_max=10.0,
        slow_twitch_ratio=0.6,
        muscle_mass=1.271640,
        tendon_slack_length=0.03,
        pennation_angle=0,
        l_mtu_opt=0.150000,
    ),

    "ercspn_l": MuscleParameters(
        f_max=2500,
        l_opt=0.12,
        v_max=10.0,
        slow_twitch_ratio=0.6,
        muscle_mass=1.271640,
        tendon_slack_length=0.03,
        pennation_angle=0,
        l_mtu_opt=0.150000,
    ),


    # ======================================================
    # intobl
    # ======================================================
    
    "intobl_r": MuscleParameters(
        f_max=2500,
        l_opt=0.12,
        v_max=10.0,
        slow_twitch_ratio=0.56,
        muscle_mass=1.271640,
        tendon_slack_length=0.03,
        pennation_angle=0,
        l_mtu_opt=0.150000,
    ),

    "intobl_l": MuscleParameters(
        f_max=2500,
        l_opt=0.12,
        v_max=10.0,
        slow_twitch_ratio=0.56,
        muscle_mass=1.271640,
        tendon_slack_length=0.03,
        pennation_angle=0,
        l_mtu_opt=0.150000,
    ),

    # ======================================================
    # extobl
    # ======================================================
    
    "extobl_r": MuscleParameters(
        f_max=900,
        l_opt=0.12,
        v_max=10.0,
        slow_twitch_ratio=0.58,
        muscle_mass=0.457790,
        tendon_slack_length=0.14,
        pennation_angle=0,
        l_mtu_opt=0.260000,
    ),

    "extobl_l": MuscleParameters(
        f_max=900,
        l_opt=0.12,
        v_max=10.0,
        slow_twitch_ratio=0.58,
        muscle_mass=0.457790,
        tendon_slack_length=0.14,
        pennation_angle=0,
        l_mtu_opt=0.260000,
    ),

}


# ==========================================================
# utility function
#
# convert dictionary-based parameters into NumPy arrays
# for vectorized computation
# ==========================================================

def get_parameter_array(muscle_names, parameter_name):

    return np.array([
        getattr(muscle_data[name], parameter_name)
        for name in muscle_names
    ])


def get_pairs():

    pairs = []

    for name in muscle_data.keys():

        if name.endswith("_r"):

            left_name = name[:-2] + "_l"

            if left_name in muscle_data:
                pairs.append((name, left_name))

    return pairs

def get_right_muscles():

    return [
        m
        for m in muscle_data.keys()
        if m.endswith("_r")
    ]
    
def get_left_muscles():

    return [
        m
        for m in muscle_data.keys()
        if m.endswith("_l")
    ]