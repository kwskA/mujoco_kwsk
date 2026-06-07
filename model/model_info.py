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

    symmetric_muscle_pairs: list = None

    default_initial_pose: str = None
    tracking_body_name: str = "pelvis"

    def get_num_muscles(self):
        """
        モデルに含まれる筋数を返す。
        """
        return len(self.muscle_names)

    def get_num_symmetric_muscles(self):
        """
        左右対称最適化時の筋数を返す。
        """

        if self.symmetric_muscle_pairs is None:
            return len(self.muscle_names)

        return len(self.symmetric_muscle_pairs)