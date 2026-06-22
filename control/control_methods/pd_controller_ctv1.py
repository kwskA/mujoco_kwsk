# control/pd_controller.py

import numpy as np

from control.base_controller import BaseController


class PDController(BaseController):
    """
    筋長に対するPD制御を行う制御器。

    use_symmetric_params=True の場合，
    左右筋ペアで同じ Kp, Kd, target_length を使用する。
    """

    def __init__(
        self,
        model,
        muscles,
        use_symmetric_params=False,
        symmetric_muscle_pairs=None,
    ):
        self.model = model
        self.muscles = list(muscles)

        self.num_muscles = len(self.muscles)

        self.use_symmetric_params = bool(use_symmetric_params)
        self.symmetric_muscle_pairs = symmetric_muscle_pairs or []

        self.actuator_ids = np.array([
            self.model.actuator(m).id
            for m in self.muscles
        ], dtype=int)

        self.tendon_ids = np.array([
            self.model.tendon(f"{m}_tendon").id
            for m in self.muscles
        ], dtype=int)

        self.Kp = None
        self.Kd = None
        self.target_length = None

        self._build_symmetric_mapping()

    def _build_symmetric_mapping(self):
        """
        対称最適化用の対応関係を作成する。

        symmetric_param_names:
            CMA-ESが直接最適化する代表筋名

        symmetric_to_full_indices:
            各筋が代表筋の何番目に対応するか
        """

        self.symmetric_param_names = list(self.muscles)
        self.symmetric_to_full_indices = np.arange(
            self.num_muscles,
            dtype=int,
        )

        if not self.use_symmetric_params:
            return

        if len(self.symmetric_muscle_pairs) == 0:
            raise ValueError(
                "use_symmetric_params=True but "
                "symmetric_muscle_pairs is empty."
            )

        muscle_to_index = {
            muscle: i
            for i, muscle in enumerate(self.muscles)
        }

        representative_names = []
        full_to_representative = {}

        for pair_index, (right_muscle, left_muscle) in enumerate(
            self.symmetric_muscle_pairs
        ):
            if right_muscle not in muscle_to_index:
                raise ValueError(
                    f"right muscle not found: {right_muscle}"
                )

            if left_muscle not in muscle_to_index:
                raise ValueError(
                    f"left muscle not found: {left_muscle}"
                )

            representative_names.append(right_muscle)

            full_to_representative[right_muscle] = pair_index
            full_to_representative[left_muscle] = pair_index

        if len(full_to_representative) != self.num_muscles:
            missing = [
                muscle
                for muscle in self.muscles
                if muscle not in full_to_representative
            ]

            raise ValueError(
                "Some muscles are not included in "
                f"symmetric_muscle_pairs: {missing}"
            )

        self.symmetric_param_names = representative_names

        self.symmetric_to_full_indices = np.array([
            full_to_representative[muscle]
            for muscle in self.muscles
        ], dtype=int)

    def _expand_symmetric_values(self, values):
        """
        対称用の代表筋パラメータを全筋分へ展開する。
        """

        values = np.asarray(values, dtype=float)

        if not self.use_symmetric_params:
            return values

        return values[self.symmetric_to_full_indices]

    def _reduce_to_symmetric_values(self, values):
        """
        全筋分の初期値から代表筋分だけ取り出す。
        """

        values = np.asarray(values, dtype=float)

        if not self.use_symmetric_params:
            return values

        muscle_to_index = {
            muscle: i
            for i, muscle in enumerate(self.muscles)
        }

        representative_indices = [
            muscle_to_index[muscle]
            for muscle in self.symmetric_param_names
        ]

        return values[representative_indices]

    def set_params_from_vector(self, x):
        """
        最適化ベクトルを Kp, Kd, target_length へ変換する。

        非対称:
            x = [Kp全筋, Kd全筋, target全筋]

        対称:
            x = [Kp代表筋, Kd代表筋, target代表筋]
            その後，全筋分へ展開する。
        """

        x = np.asarray(x, dtype=float)

        n = self.get_effective_num_muscles()
        expected_dim = 3 * n

        if x.size != expected_dim:
            raise ValueError(
                f"Invalid parameter size: got {x.size}, "
                f"expected {expected_dim}"
            )

        Kp_base = x[:n]
        Kd_base = x[n:2*n]
        target_base = x[2*n:3*n]

        self.Kp = self._expand_symmetric_values(Kp_base)
        self.Kd = self._expand_symmetric_values(Kd_base)
        self.target_length = self._expand_symmetric_values(target_base)

    def compute_ctrl(self, data, state_r=None, state_l=None):
        """
        筋長と筋速度から制御入力を計算する。
        """

        if self.Kp is None or self.Kd is None or self.target_length is None:
            raise RuntimeError(
                "PDController parameters are not set. "
                "Call set_params_from_vector() first."
            )

        ctrl = np.zeros(self.model.nu)

        current_length = np.array([
            data.ten_length[tendon_id]
            for tendon_id in self.tendon_ids
        ])

        current_velocity = np.array([
            data.ten_velocity[tendon_id]
            for tendon_id in self.tendon_ids
        ])

        u = (
            self.Kp * (
                current_length - self.target_length
            )
            + self.Kd * current_velocity
        )

        u = np.clip(u, 0.0, 1.0)

        ctrl[self.actuator_ids] = u

        return ctrl

    def get_effective_num_muscles(self):
        """
        最適化対象として扱う筋数を返す。

        非対称:
            実筋数

        対称:
            左右ペアの代表筋数
        """

        if self.use_symmetric_params:
            return len(self.symmetric_param_names)

        return self.num_muscles

    def get_expected_param_dim(self):
        """
        必要な最適化パラメータ数を返す。
        """

        return 3 * self.get_effective_num_muscles()

    def make_initial_params(
        self,
        init_Kp=10.0,
        init_Kd=2.0,
        init_target_length=None,
    ):
        """
        初期パラメータを生成する。

        非対称:
            全筋分の初期値を返す

        対称:
            代表筋分のみの初期値を返す
        """

        if init_target_length is None:
            raise ValueError(
                "init_target_length is required"
            )

        init_target_length = np.asarray(
            init_target_length,
            dtype=float,
        )

        if init_target_length.size != self.num_muscles:
            raise ValueError(
                f"Invalid init_target_length size: "
                f"got {init_target_length.size}, "
                f"expected {self.num_muscles}"
            )

        if self.use_symmetric_params:
            n = self.get_effective_num_muscles()

            target_base = self._reduce_to_symmetric_values(
                init_target_length
            )

            return np.concatenate([
                np.full(n, init_Kp),
                np.full(n, init_Kd),
                target_base,
            ])

        return np.concatenate([
            np.full(self.num_muscles, init_Kp),
            np.full(self.num_muscles, init_Kd),
            init_target_length,
        ])