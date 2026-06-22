from dataclasses import dataclass
import numpy as np


# ==========================================================
# Muscle parameter class
# ==========================================================

@dataclass
class MuscleParameters:

    # maximum isometric force [N]
    # cf. gait10dof18musc.osim
    f_max: float

    # optimal fiber length [m]
    # cf. gait10dof18musc.osim
    l_opt: float

    # maximum contraction velocity [l_opt / s]
    # cf. gait10dof18musc.osim
    v_max: float

    # slow twitch fiber ratio [0-1]
    # cf. Documents/OpenSim/4.5/Models/Gait10dof18musc/subject01_metabolics.osim
    slow_twitch_ratio: float

    # muscle mass [kg]
    # muscle_mass [kg] = (f_max / specific_tension) * density * length_opt
    # specific_tension = 0.25e6 [N/m^2]
    # density = 1059.7 [kg/m^3]
    # https://simtk.org/api_docs/opensim/api_docs/classOpenSim_1_1Umberger2010MuscleMetabolicsProbe__MetabolicMuscleParameter.html
    muscle_mass: float

    # tendon slack length [m]
    # cf. gait10dof18musc.osim
    tendon_slack_length: float

    # pennation angle at optimal fiber length [rad]
    # cf. gait10dof18musc.osim
    pennation_angle: float

    # optimal MTU length [m]
    #
    # used for:
    # l_ce_norm ≈ l_MTU / l_MTU_opt
    #
    # l_mtu_opt = tendon_slack_length + l_opt*cos(pnnation_angle)
    l_mtu_opt: float


# ==========================================================
# gait10dof18musc muscle parameters
#
# IMPORTANT:
# right and left muscles are intentionally stored separately
# because future asymmetric parameter tuning may be required.
# ==========================================================

muscle_data = {

    # ======================================================
    # hamstrings
    # ======================================================

    "hamstrings_r": MuscleParameters(
        f_max=2700,
        l_opt=0.109,
        v_max=10.0,
        slow_twitch_ratio=0.49,
        muscle_mass=1.24747884,
        tendon_slack_length=0.326,
        pennation_angle=0.0,
        l_mtu_opt=0.435,
    ),

    "hamstrings_l": MuscleParameters(
        f_max=2700,
        l_opt=0.109,
        v_max=10.0,
        slow_twitch_ratio=0.49,
        muscle_mass=1.24747884,
        tendon_slack_length=0.326,
        pennation_angle=0.0,
        l_mtu_opt=0.435,
    ),

    # ======================================================
    # bifemsh
    # ======================================================

    "bifemsh_r": MuscleParameters(
        f_max=804,
        l_opt=0.173,
        v_max=10.0,
        slow_twitch_ratio=0.53,
        muscle_mass=0.58958317,
        tendon_slack_length=0.089,
        pennation_angle=0.40142573,
        l_mtu_opt=0.2482473395,
    ),

    "bifemsh_l": MuscleParameters(
        f_max=804,
        l_opt=0.173,
        v_max=10.0,
        slow_twitch_ratio=0.53,
        muscle_mass=0.58958317,
        tendon_slack_length=0.089,
        pennation_angle=0.40142573,
        l_mtu_opt=0.2482473395,
    ),

    # ======================================================
    # glut_max
    # ======================================================

    "glut_max_r": MuscleParameters(
        f_max=1944,
        l_opt=0.147,
        v_max=10.0,
        slow_twitch_ratio=0.55,
        muscle_mass=1.2113134,
        tendon_slack_length=0.127,
        pennation_angle=0.0,
        l_mtu_opt=0.274,
    ),

    "glut_max_l": MuscleParameters(
        f_max=1944,
        l_opt=0.147,
        v_max=10.0,
        slow_twitch_ratio=0.55,
        muscle_mass=1.2113134,
        tendon_slack_length=0.127,
        pennation_angle=0.0,
        l_mtu_opt=0.274,
    ),

    # ======================================================
    # iliopsoas
    # ======================================================

    "iliopsoas_r": MuscleParameters(
        f_max=2342,
        l_opt=0.1,
        v_max=10.0,
        slow_twitch_ratio=0.50,
        muscle_mass=0.99272696,
        tendon_slack_length=0.16,
        pennation_angle=0.13962634,
        l_mtu_opt=0.2590268069,
    ),

    "iliopsoas_l": MuscleParameters(
        f_max=2342,
        l_opt=0.1,
        v_max=10.0,
        slow_twitch_ratio=0.50,
        muscle_mass=0.99272696,
        tendon_slack_length=0.16,
        pennation_angle=0.13962634,
        l_mtu_opt=0.2590268069,
    ),

    # ======================================================
    # rect_fem
    # ======================================================

    "rect_fem_r": MuscleParameters(
        f_max=1169,
        l_opt=0.114,
        v_max=10.0,
        slow_twitch_ratio=0.39,
        muscle_mass=0.56488792,
        tendon_slack_length=0.31,
        pennation_angle=0.08726646,
        l_mtu_opt=0.4235661956,
    ),

    "rect_fem_l": MuscleParameters(
        f_max=1169,
        l_opt=0.114,
        v_max=10.0,
        slow_twitch_ratio=0.39,
        muscle_mass=0.56488792,
        tendon_slack_length=0.31,
        pennation_angle=0.08726646,
        l_mtu_opt=0.4235661956,
    ),

    # ======================================================
    # vasti
    # ======================================================

    "vasti_r": MuscleParameters(
        f_max=5000,
        l_opt=0.087,
        v_max=10.0,
        slow_twitch_ratio=0.50,
        muscle_mass=1.843878,
        tendon_slack_length=0.136,
        pennation_angle=0.05235988,
        l_mtu_opt=0.2228807695,
    ),

    "vasti_l": MuscleParameters(
        f_max=5000,
        l_opt=0.087,
        v_max=10.0,
        slow_twitch_ratio=0.50,
        muscle_mass=1.843878,
        tendon_slack_length=0.136,
        pennation_angle=0.05235988,
        l_mtu_opt=0.2228807695,
    ),

    # ======================================================
    # gastroc
    # ======================================================

    "gastroc_r": MuscleParameters(
        f_max=2500,
        l_opt=0.06,
        v_max=10.0,
        slow_twitch_ratio=0.54,
        muscle_mass=0.63582,
        tendon_slack_length=0.39,
        pennation_angle=0.29670597,
        l_mtu_opt=0.4435530664,
    ),

    "gastroc_l": MuscleParameters(
        f_max=2500,
        l_opt=0.06,
        v_max=10.0,
        slow_twitch_ratio=0.54,
        muscle_mass=0.63582,
        tendon_slack_length=0.39,
        pennation_angle=0.29670597,
        l_mtu_opt=0.4435530664,
    ),

    # ======================================================
    # soleus
    # ======================================================

    "soleus_r": MuscleParameters(
        f_max=5137,
        l_opt=0.05,
        v_max=10.0,
        slow_twitch_ratio=0.80,
        muscle_mass=1.08873578,
        tendon_slack_length=0.25,
        pennation_angle=0.43633231,
        l_mtu_opt=0.2953153894,
    ),

    "soleus_l": MuscleParameters(
        f_max=5137,
        l_opt=0.05,
        v_max=10.0,
        slow_twitch_ratio=0.80,
        muscle_mass=1.08873578,
        tendon_slack_length=0.25,
        pennation_angle=0.43633231,
        l_mtu_opt=0.2953153894,
    ),

    # ======================================================
    # tib_ant
    # ======================================================

    "tib_ant_r": MuscleParameters(
        f_max=3000,
        l_opt=0.098,
        v_max=10.0,
        slow_twitch_ratio=0.70,
        muscle_mass=1.2462072,
        tendon_slack_length=0.223,
        pennation_angle=0.08726646,
        l_mtu_opt=0.3206270804,
    ),

    "tib_ant_l": MuscleParameters(
        f_max=3000,
        l_opt=0.098,
        v_max=10.0,
        slow_twitch_ratio=0.70,
        muscle_mass=1.2462072,
        tendon_slack_length=0.223,
        pennation_angle=0.08726646,
        l_mtu_opt=0.3206270804,
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
# muscle order used in MuJoCo actuator indexing
# ==========================================================

# muscle_names = [
#     "hamstrings_r",
#     "bifemsh_r",
#     "glut_max_r",
#     "iliopsoas_r",
#     "rect_fem_r",
#     "vasti_r",
#     "gastroc_r",
#     "soleus_r",
#     "tib_ant_r",

#     "hamstrings_l",
#     "bifemsh_l",
#     "glut_max_l",
#     "iliopsoas_l",
#     "rect_fem_l",
#     "vasti_l",
#     "gastroc_l",
#     "soleus_l",
#     "tib_ant_l",
# ]


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