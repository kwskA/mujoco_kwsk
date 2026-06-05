# control/extended_p_controller.py

from control.control_methods.p_controller import PController


class ExtendedPController(PController):
    """
    拡張P制御用クラス。

    現時点ではP制御と同じ。
    今後、
        筋力フィードバック
        COPフィードバック
        COMフィードバック
    などを追加する。
    """

    pass