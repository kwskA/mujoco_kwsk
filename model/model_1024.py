# model/gait1024.py

from model.model_info import ModelInfo


GAIT1024_MUSCLES = [
    "hamstrings_r",
    "bifemsh_r",
    "glut_max_r",
    "iliopsoas_r",
    "rect_fem_r",
    "vasti_r",
    "gastroc_r",
    "soleus_r",
    "tib_ant_r",
    "ercspn_r",
    "intobl_r",
    "extobl_r",

    "hamstrings_l",
    "bifemsh_l",
    "glut_max_l",
    "iliopsoas_l",
    "rect_fem_l",
    "vasti_l",
    "gastroc_l",
    "soleus_l",
    "tib_ant_l",
    "ercspn_l",
    "intobl_l",
    "extobl_l",
]


GAIT1024_SYMMETRIC_MUSCLE_PAIRS = [
    ("hamstrings_r", "hamstrings_l"),
    ("bifemsh_r", "bifemsh_l"),
    ("glut_max_r", "glut_max_l"),
    ("iliopsoas_r", "iliopsoas_l"),
    ("rect_fem_r", "rect_fem_l"),
    ("vasti_r", "vasti_l"),
    ("gastroc_r", "gastroc_l"),
    ("soleus_r", "soleus_l"),
    ("tib_ant_r", "tib_ant_l"),
    ("ercspn_r", "ercspn_l"),
    ("intobl_r", "intobl_l"),
    ("extobl_r", "extobl_l"),
]


def create_model_info():
    """
    10自由度24筋モデルの情報を作成する。
    """

    return ModelInfo(
        name="gait1024",
        model_path="model/1024model.xml",
        muscle_names=GAIT1024_MUSCLES,
        symmetric_muscle_pairs=GAIT1024_SYMMETRIC_MUSCLE_PAIRS,
        default_initial_pose="EarlyStance",
        tracking_body_name="pelvis",
    )