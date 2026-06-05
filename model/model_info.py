# model/model_info.py

from dataclasses import dataclass


@dataclass
class ModelInfo:
    """
    使用するMuJoCoモデルに関する情報をまとめるクラス。
    """

    name: str
    model_path: str
    muscle_names: list
    default_initial_pose: str = None
    tracking_body_name: str = "pelvis"

    def get_num_muscles(self):
        """
        モデルに含まれる筋数を返す。
        """
        return len(self.muscle_names)