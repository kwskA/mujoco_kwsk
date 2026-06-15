from dataclasses import dataclass
import numpy as np


@dataclass
class MuscleParameters:
    f_max: float
    l_opt: float
    v_max: float
    slow_twitch_ratio: float
    muscle_mass: float
    tendon_slack_length: float
    pennation_angle: float
    l_mtu_opt: float


muscle_data = {
    # abd / add
    "abd_r": MuscleParameters(4304.64, 0.0845, 10.0, 0.50, 1.54328839, 0.0535, 0.0, 0.138),
    "abd_l": MuscleParameters(4172.08, 0.0845, 10.0, 0.50, 1.49577047, 0.0535, 0.0, 0.138),

    "add_r": MuscleParameters(3650.15, 0.0870, 10.0, 0.50, 1.34609932, 0.0600, 0.0, 0.147),
    "add_l": MuscleParameters(3440.67, 0.0870, 10.0, 0.50, 1.26886254, 0.0600, 0.0, 0.147),

    # hamstrings
    "hamstrings_r": MuscleParameters(3547.72, 0.109, 10.0, 0.49, 1.63920761, 0.326, 0.0, 0.435),
    "hamstrings_l": MuscleParameters(3356.57, 0.109, 10.0, 0.49, 1.55089739, 0.326, 0.0, 0.435),

    # bifemsh
    "bifemsh_r": MuscleParameters(504.584, 0.173, 10.0, 0.53, 0.36997954, 0.089, 0.40142573, 0.2482473395),
    "bifemsh_l": MuscleParameters(503.252, 0.173, 10.0, 0.53, 0.36900275, 0.089, 0.40142573, 0.2482473395),

    # glut_max
    "glut_max_r": MuscleParameters(3259.98, 0.147, 10.0, 0.55, 2.03155651, 0.127, 0.0, 0.274),
    "glut_max_l": MuscleParameters(3279.09, 0.147, 10.0, 0.55, 2.04346488, 0.127, 0.0, 0.274),

    # iliopsoas
    "iliopsoas_r": MuscleParameters(2531.45, 0.1, 10.0, 0.50, 1.07304733, 0.16, 0.13962634, 0.2590268069),
    "iliopsoas_l": MuscleParameters(2569.96, 0.1, 10.0, 0.50, 1.08937105, 0.16, 0.13962634, 0.2590268069),

    # rect_fem
    "rect_fem_r": MuscleParameters(1327.81, 0.114, 10.0, 0.39, 0.64163020, 0.31, 0.08726646, 0.4235661956),
    "rect_fem_l": MuscleParameters(1343.37, 0.114, 10.0, 0.39, 0.64914913, 0.31, 0.08726646, 0.4235661956),

    # vasti
    "vasti_r": MuscleParameters(9473.54, 0.087, 10.0, 0.50, 3.49328709, 0.136, 0.05235988, 0.2228807695),
    "vasti_l": MuscleParameters(9491.74, 0.087, 10.0, 0.50, 3.49999936, 0.136, 0.05235988, 0.2228807695),

    # gastroc
    "gastroc_r": MuscleParameters(3836.52, 0.06, 10.0, 0.54, 0.97575530, 0.39, 0.29670597, 0.4435530664),
    "gastroc_l": MuscleParameters(4364.98, 0.06, 10.0, 0.54, 1.11014283, 0.39, 0.29670597, 0.4435530664),

    # soleus
    "soleus_r": MuscleParameters(6675.41, 0.05, 10.0, 0.80, 1.41491516, 0.25, 0.43633231, 0.2953153894),
    "soleus_l": MuscleParameters(6968.05, 0.05, 10.0, 0.80, 1.47694787, 0.25, 0.43633231, 0.2953153894),

    # tib_ant
    "tib_ant_r": MuscleParameters(2050.78, 0.098, 10.0, 0.70, 0.85174494, 0.223, 0.08726646, 0.3206270804),
    "tib_ant_l": MuscleParameters(1996.3, 0.098, 10.0, 0.70, 0.82912626, 0.223, 0.08726646, 0.3206270804),
}


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