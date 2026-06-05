# model/gait1018.py

from model.model_info import ModelInfo


GAIT1018_MUSCLES = [
    "hamstrings_r",
    "bifemsh_r",
    "glut_max_r",
    "iliopsoas_r",
    "rect_fem_r",
    "vasti_r",
    "gastroc_r",
    "soleus_r",
    "tib_ant_r",

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


def create_model_info():
    """
    10自由度18筋モデルの情報を作成する。
    """

    return ModelInfo(
        name="gait1018",
        model_path="model/1018model.xml",
        muscle_names=GAIT1018_MUSCLES,
        default_initial_pose="EarlyStance",
        tracking_body_name="pelvis",
    )