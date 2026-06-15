# model/gait1422.py

from model.model_info import ModelInfo


GAIT1422_MUSCLES = [
    "abd_r",
    "add_r",
    "hamstrings_r",
    "bifemsh_r",
    "glut_max_r",
    "iliopsoas_r",
    "rect_fem_r",
    "vasti_r",
    "gastroc_r",
    "soleus_r",
    "tib_ant_r",

    "abd_l",
    "add_l",
    "hamstrings_l",
    "bifemsh_l",
    "glut_max_l",
    "iliopsoas_l",
    "rect_fem_l",
    "vasti_l",
    "gastroc_l",
    "soleus_l",
    "tib_ant_l",
]


GAIT1422_SYMMETRIC_MUSCLE_PAIRS = [
    ("abd_r", "abd_l"),
    ("add_r", "add_l"),
    ("hamstrings_r", "hamstrings_l"),
    ("bifemsh_r", "bifemsh_l"),
    ("glut_max_r", "glut_max_l"),
    ("iliopsoas_r", "iliopsoas_l"),
    ("rect_fem_r", "rect_fem_l"),
    ("vasti_r", "vasti_l"),
    ("gastroc_r", "gastroc_l"),
    ("soleus_r", "soleus_l"),
    ("tib_ant_r", "tib_ant_l"),
]


def create_model_info():
    """
    22筋モデルの情報を作成する。
    """

    return ModelInfo(
        name="gait1422",
        model_path="model/1422model.xml",
        muscle_names=GAIT1422_MUSCLES,
        symmetric_muscle_pairs=GAIT1422_SYMMETRIC_MUSCLE_PAIRS,
        default_initial_pose="EarlyStance",
        tracking_body_name="pelvis",
    )