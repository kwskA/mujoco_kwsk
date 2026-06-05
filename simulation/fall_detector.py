# simulation/fall_detector.py


class COMHeightFallDetector:
    """
    COM高さに基づいて転倒判定を行うクラス。

    初期COM高さの threshold_ratio 倍を下回った場合に転倒と判定する。
    """

    def __init__(
        self,
        com_init_height,
        threshold_ratio=0.9,
    ):
        """
        初期COM高さと転倒判定しきい値を保存する。
        """

        self.com_init_height = float(com_init_height)
        self.threshold_ratio = float(threshold_ratio)

        self.threshold_height = (
            self.com_init_height
            * self.threshold_ratio
        )

    def is_fallen(self, model, data, step=None):
        """
        現在のCOM高さがしきい値を下回っているか判定する。
        """

        com_z = float(data.subtree_com[0][2])

        return com_z < self.threshold_height

    def get_info(self):
        """
        転倒判定条件の情報を返す。
        """

        return {
            "type": "COMHeightFallDetector",
            "com_init_height": self.com_init_height,
            "threshold_ratio": self.threshold_ratio,
            "threshold_height": self.threshold_height,
        }