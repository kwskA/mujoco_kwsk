# model/gait2354.py

from model.model_info import ModelInfo


GAIT2354_MUSCLES = [
    # --- 右下肢 ---
    "glut_med1_r",
    "glut_med2_r",
    "glut_med3_r",
    "bifemlh_r",
    "bifemsh_r",
    "sar_r",
    "add_mag2_r",
    "tfl_r",
    "pect_r",
    "grac_r",
    "glut_max1_r",
    "glut_max2_r",
    "glut_max3_r",
    "iliacus_r",
    "psoas_r",
    "quad_fem_r",
    "gem_r",
    "peri_r",
    "rect_fem_r",
    "vas_int_r",
    "med_gas_r",
    "soleus_r",
    "tib_post_r",
    "tib_ant_r",

    # --- 左下肢 ---
    "glut_med1_l",
    "glut_med2_l",
    "glut_med3_l",
    "bifemlh_l",
    "bifemsh_l",
    "sar_l",
    "add_mag2_l",
    "tfl_l",
    "pect_l",
    "grac_l",
    "glut_max1_l",
    "glut_max2_l",
    "glut_max3_l",
    "iliacus_l",
    "psoas_l",
    "quad_fem_l",
    "gem_l",
    "peri_l",
    "rect_fem_l",
    "vas_int_l",
    "med_gas_l",
    "soleus_l",
    "tib_post_l",
    "tib_ant_l",
]


GAIT2354_SYMMETRIC_MUSCLE_PAIRS = [
    ("glut_med1_r", "glut_med1_l"),
    ("glut_med2_r", "glut_med2_l"),
    ("glut_med3_r", "glut_med3_l"),

    ("bifemlh_r", "bifemlh_l"),
    ("bifemsh_r", "bifemsh_l"),

    ("sar_r", "sar_l"),
    ("add_mag2_r", "add_mag2_l"),
    ("tfl_r", "tfl_l"),
    ("pect_r", "pect_l"),
    ("grac_r", "grac_l"),

    ("glut_max1_r", "glut_max1_l"),
    ("glut_max2_r", "glut_max2_l"),
    ("glut_max3_r", "glut_max3_l"),

    ("iliacus_r", "iliacus_l"),
    ("psoas_r", "psoas_l"),

    ("quad_fem_r", "quad_fem_l"),
    ("gem_r", "gem_l"),
    ("peri_r", "peri_l"),

    ("rect_fem_r", "rect_fem_l"),
    ("vas_int_r", "vas_int_l"),

    ("med_gas_r", "med_gas_l"),
    ("soleus_r", "soleus_l"),

    ("tib_post_r", "tib_post_l"),
    ("tib_ant_r", "tib_ant_l"),
]


def create_model_info():
    """
    23自由度54筋モデルの情報を作成する。
    """

    return ModelInfo(
        name="2354",
        model_path="model/2354model.xml",
        muscle_names=GAIT2354_MUSCLES,
        symmetric_muscle_pairs=GAIT2354_SYMMETRIC_MUSCLE_PAIRS,
        default_initial_pose="EarlyStance",
        tracking_body_name="pelvis",
    )