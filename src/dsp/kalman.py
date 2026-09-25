from __future__ import annotations

import numpy as np

EPSILON = 1e-12


class DelayKalmanFilter:
    """
    Stateful Scalar Kalman Filter for tracking relative delay between M1 and M2.

    State:
        x_k = relative delay between M1 and M2 in samples.

    Measurement:
        z_k = raw relative delay observation from GCC-PHAT in samples.

    Sign Convention:
        - Positive delay (+tau > 0): M2 is delayed relative to M1.
        - Negative delay (-tau < 0): M1 is delayed relative to M2.
        - Zero delay (tau == 0): Signals arrive at M1 and M2 simultaneously.

    Kalman Equations:
        Prediction:
            x_pred = x
            P_pred = P + Q
        Innovation:
            y = z - x_pred
            S = P_pred + R
            K = P_pred / S
        Update:
            x = x_pred + K * y
            P = (1 - K) * P_pred
    """

    def __init__(
        self,
        initial_delay: float = 0.0,
        initial_covariance: float = 1.0,
        process_noise: float = 0.01,
        measurement_noise: float = 1.0,
    ) -> None:
        self._validate_float(initial_delay, "initial_delay")
        self._validate_non_negative(initial_covariance, "initial_covariance")
        self._validate_non_negative(process_noise, "process_noise")
        self._validate_positive(measurement_noise, "measurement_noise")

        self.initial_delay = float(initial_delay)
        self.initial_covariance = float(initial_covariance)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)

        self._x = self.initial_delay
        self._P = self.initial_covariance

    @property
    def current_state(self) -> float:
        """Return current estimated relative delay in samples."""
        return float(self._x)

    @property
    def current_covariance(self) -> float:
        """Return current state error covariance."""
        return float(self._P)

    def update(self, measurement: float | int) -> float:
        """
        Incorporate a new raw GCC-PHAT relative delay measurement and update state.

        Parameters
        ----------
        measurement : float | int
            Raw relative delay observation z_k in samples.

        Returns
        -------
        float
            Updated filtered relative delay estimate x_k in samples.
        """
        self._validate_float(measurement, "measurement")
        z = float(measurement)

        # 1. Prediction step
        x_pred = self._x
        P_pred = self._P + self.process_noise

        # 2. Measurement residual and innovation
        y = z - x_pred
        S = P_pred + self.measurement_noise
        if abs(S) < EPSILON:
            S = EPSILON

        # 3. Kalman Gain
        K = P_pred / S

        # 4. State update
        self._x = float(x_pred + K * y)
        self._P = float((1.0 - K) * P_pred)

        if not np.isfinite(self._x) or not np.isfinite(self._P):
            raise RuntimeError("Kalman filter produced non-finite state or covariance")

        return self._x

    def reset(
        self,
        delay: float | int | None = None,
        covariance: float | int | None = None,
    ) -> None:
        """
        Reset state and covariance to configured or custom values.
        """
        if delay is not None:
            self._validate_float(delay, "delay")
            self._x = float(delay)
        else:
            self._x = self.initial_delay

        if covariance is not None:
            self._validate_non_negative(covariance, "covariance")
            self._P = float(covariance)
        else:
            self._P = self.initial_covariance

    @staticmethod
    def _validate_float(val: float | int, name: str) -> None:
        if not isinstance(val, (int, float, np.number)):
            raise TypeError(f"{name} must be a number, got {type(val).__name__}")
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val}")

    @staticmethod
    def _validate_non_negative(val: float | int, name: str) -> None:
        DelayKalmanFilter._validate_float(val, name)
        if val < 0:
            raise ValueError(f"{name} must be >= 0, got {val}")

    @staticmethod
    def _validate_positive(val: float | int, name: str) -> None:
        DelayKalmanFilter._validate_float(val, name)
        if val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")


def filter_delay_sequence(
    measurements: Sequence[float | int],
    initial_delay: float = 0.0,
    initial_covariance: float = 1.0,
    process_noise: float = 0.01,
    measurement_noise: float = 1.0,
) -> np.ndarray:
    """
    Convenience function to run DelayKalmanFilter over a sequence of delay observations.

    Returns 1-D float32 array of filtered delay estimates.
    """
    kf = DelayKalmanFilter(
        initial_delay=initial_delay,
        initial_covariance=initial_covariance,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
    )
    filtered = [kf.update(m) for m in measurements]
    return np.array(filtered, dtype=np.float32)
