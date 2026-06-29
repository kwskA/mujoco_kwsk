# control/control_methods/pd_ff_controller.py

from collections import deque

import numpy as np

from control.base_controller import BaseController


class PDFFController(BaseController):
    """
    筋長に対するPD + FeedForward制御を行う制御器。

    u = activation_ff
        + Kp * (delayed_length - target_length)
        + Kd * delayed_velocity

    sensor_delay_ms により，筋長・筋速度のセンサ情報を遅延させる。
    """

    def __init__(
        self,
        model,
        muscles,
        use_symmetric_params=False,
        symmetric_muscle_pairs=None,
        use_gait_states=False,
        gait_states=None,
        sensor_delay_steps=0.0,
        noise_std_length=0.0,
        noise_std_velocity=0.0,
        random_seed=None,
    ):
        self.model = model
        self.muscles = list(muscles)

        self.num_muscles = len(self.muscles)

        self.use_symmetric_params = bool(use_symmetric_params)
        self.symmetric_muscle_pairs = symmetric_muscle_pairs or []

        self.use_gait_states = bool(use_gait_states)

        if gait_states is None:
            gait_states = [
                "EarlyStance",
                "LateStance",
                "Liftoff",
                "Swing",
                "Landing",
            ]

        self.gait_states = list(gait_states)
        self.num_gait_states = len(self.gait_states)

        self.state_to_index = {
            state: i
            for i, state in enumerate(self.gait_states)
        }

        self.actuator_ids = np.array([
            self.model.actuator(m).id
            for m in self.muscles
        ], dtype=int)

        self.tendon_ids = np.array([
            self.model.tendon(f"{m}_tendon").id
            for m in self.muscles
        ], dtype=int)

        self.activation_ff = None
        self.Kp = None
        self.Kd = None
        self.target_length = None

        self.sensor_delay_steps = int(sensor_delay_steps)

        self.length_buffer = deque(
            maxlen=self.sensor_delay_steps + 1
        )
        self.velocity_buffer = deque(
            maxlen=self.sensor_delay_steps + 1
        )

        self.noise_std_length = float(noise_std_length)
        self.noise_std_velocity = float(noise_std_velocity)
        self.rng = np.random.default_rng(random_seed)

        self._build_symmetric_mapping()

    def reset(self):
        """
        シミュレーション開始時に遅延バッファをリセットする。
        """
        self.length_buffer.clear()
        self.velocity_buffer.clear()

    def _build_symmetric_mapping(self):
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
        values = np.asarray(values, dtype=float)

        if not self.use_symmetric_params:
            return values

        return values[self.symmetric_to_full_indices]

    def _reduce_to_symmetric_values(self, values):
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
        x = np.asarray(x, dtype=float)

        n = self.get_effective_num_muscles()

        if self.use_gait_states:
            expected_dim = 4 * n * self.num_gait_states

            if x.size != expected_dim:
                raise ValueError(
                    f"Invalid parameter size: got {x.size}, "
                    f"expected {expected_dim}"
                )

            block = n * self.num_gait_states

            activation_ff_base = x[:block].reshape(
                self.num_gait_states,
                n,
            )
            Kp_base = x[block:2 * block].reshape(
                self.num_gait_states,
                n,
            )
            Kd_base = x[2 * block:3 * block].reshape(
                self.num_gait_states,
                n,
            )
            target_base = x[3 * block:4 * block].reshape(
                self.num_gait_states,
                n,
            )

            self.activation_ff = np.vstack([
                self._expand_symmetric_values(
                    activation_ff_base[state_index]
                )
                for state_index in range(self.num_gait_states)
            ])

            self.Kp = np.vstack([
                self._expand_symmetric_values(Kp_base[state_index])
                for state_index in range(self.num_gait_states)
            ])

            self.Kd = np.vstack([
                self._expand_symmetric_values(Kd_base[state_index])
                for state_index in range(self.num_gait_states)
            ])

            self.target_length = np.vstack([
                self._expand_symmetric_values(target_base[state_index])
                for state_index in range(self.num_gait_states)
            ])

            return

        expected_dim = 4 * n

        if x.size != expected_dim:
            raise ValueError(
                f"Invalid parameter size: got {x.size}, "
                f"expected {expected_dim}"
            )

        activation_ff_base = x[:n]
        Kp_base = x[n:2 * n]
        Kd_base = x[2 * n:3 * n]
        target_base = x[3 * n:4 * n]

        self.activation_ff = self._expand_symmetric_values(
            activation_ff_base
        )
        self.Kp = self._expand_symmetric_values(Kp_base)
        self.Kd = self._expand_symmetric_values(Kd_base)
        self.target_length = self._expand_symmetric_values(target_base)

    def _get_state_index_for_muscle(self, muscle_name, state_r, state_l):
        if muscle_name.endswith("_r"):
            state = state_r
        elif muscle_name.endswith("_l"):
            state = state_l
        else:
            state = state_r

        if state not in self.state_to_index:
            return 0

        return self.state_to_index[state]

    def _get_delayed_sensor_values(
        self,
        current_length,
        current_velocity,
    ):
        self.length_buffer.append(current_length.copy())
        self.velocity_buffer.append(current_velocity.copy())

        if len(self.length_buffer) <= self.sensor_delay_steps:
            delayed_length = current_length.copy()
            delayed_velocity = current_velocity.copy()
        else:
            delayed_length = self.length_buffer[0].copy()
            delayed_velocity = self.velocity_buffer[0].copy()

        if self.noise_std_length > 0.0:
            delayed_length += self.rng.normal(
                0.0,
                self.noise_std_length,
                size=delayed_length.shape,
            )

        if self.noise_std_velocity > 0.0:
            delayed_velocity += self.rng.normal(
                0.0,
                self.noise_std_velocity,
                size=delayed_velocity.shape,
            )

        return delayed_length, delayed_velocity

    def compute_ctrl(self, data, state_r=None, state_l=None):
        if (
            self.activation_ff is None
            or self.Kp is None
            or self.Kd is None
            or self.target_length is None
        ):
            raise RuntimeError(
                "PDFFController parameters are not set. "
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

        delayed_length, delayed_velocity = self._get_delayed_sensor_values(
            current_length=current_length,
            current_velocity=current_velocity,
        )

        if self.use_gait_states:
            if state_r is None or state_l is None:
                raise ValueError(
                    "state_r and state_l are required "
                    "when use_gait_states=True."
                )

            activation_ff = np.zeros(self.num_muscles)
            Kp = np.zeros(self.num_muscles)
            Kd = np.zeros(self.num_muscles)
            target_length = np.zeros(self.num_muscles)

            for muscle_index, muscle_name in enumerate(self.muscles):
                state_index = self._get_state_index_for_muscle(
                    muscle_name,
                    state_r,
                    state_l,
                )

                activation_ff[muscle_index] = self.activation_ff[
                    state_index,
                    muscle_index,
                ]

                Kp[muscle_index] = self.Kp[
                    state_index,
                    muscle_index,
                ]

                Kd[muscle_index] = self.Kd[
                    state_index,
                    muscle_index,
                ]

                target_length[muscle_index] = self.target_length[
                    state_index,
                    muscle_index,
                ]

        else:
            activation_ff = self.activation_ff
            Kp = self.Kp
            Kd = self.Kd
            target_length = self.target_length

        u = (
            activation_ff
            + Kp * (
                delayed_length - target_length
            )
            + Kd * delayed_velocity
        )

        u = np.clip(u, 0.0, 1.0)

        ctrl[self.actuator_ids] = u

        return ctrl

    def get_effective_num_muscles(self):
        if self.use_symmetric_params:
            return len(self.symmetric_param_names)

        return self.num_muscles

    def get_expected_param_dim(self):
        if self.use_gait_states:
            return (
                4
                * self.get_effective_num_muscles()
                * self.num_gait_states
            )

        return 4 * self.get_effective_num_muscles()

    def make_initial_params(
        self,
        init_activation_ff=0.05,
        init_Kp=10.0,
        init_Kd=2.0,
        init_target_length=None,
    ):
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
        else:
            n = self.num_muscles
            target_base = init_target_length

        if self.use_gait_states:
            activation_ff_all = np.full(
                self.num_gait_states * n,
                init_activation_ff,
            )

            Kp_all = np.full(
                self.num_gait_states * n,
                init_Kp,
            )

            Kd_all = np.full(
                self.num_gait_states * n,
                init_Kd,
            )

            target_all = np.tile(
                target_base,
                self.num_gait_states,
            )

            return np.concatenate([
                activation_ff_all,
                Kp_all,
                Kd_all,
                target_all,
            ])

        return np.concatenate([
            np.full(n, init_activation_ff),
            np.full(n, init_Kp),
            np.full(n, init_Kd),
            target_base,
        ])