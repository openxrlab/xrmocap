import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanJointTracker(object):
    """This class represents the internal state of individual tracked objects
    observed as joint set.

    state model: x, y, z, dx, dy, dz observation model: x, y, z As implemented
    in https://github.com/abewley/sort, but with some modifications.
    """

    def __init__(self, kps3d: np.ndarray):
        """Initialises a tracker using initial body keypoints3d.

        Args:
            kps3d (np.ndarray): initial body keypoints3d, in shape
                (n_kps3d, 3).
        """
        # define constant velocity model
        self.n_kps3d = kps3d.shape[0]
        self.kf = []
        state_model = np.eye(6)
        state_model[:3, 3:] = np.eye(3)
        observation_model = np.eye(6)[:3]
        for i in range(self.n_kps3d):
            kf = KalmanFilter(dim_x=6, dim_z=3)
            kf.F = state_model
            kf.H = observation_model
            kf.P[3:, 3:] *= 100.
            kf.P *= 10.
            kf.Q[3:, 3:] *= 0.01
            kf.x[:3] = np.expand_dims(kps3d[i], -1)
            self.kf.append(kf)

    def predict(self):
        """Advances the state vector and returns the predicted body keypoints3d
        estimate."""
        for i in range(self.n_kps3d):
            self.kf[i].predict()

    def update(self, kps3d: np.ndarray):
        """Updates the state vector with observed body keypoints3d.

        Args:
            kps3d (np.ndarray): The measurement 3d keypoints.
        """
        for i in range(self.n_kps3d):
            self.kf[i].update(kps3d[i].reshape(-1, 1))

    def get_update(self) -> list:
        """Returns the new estimate based on measurement `z`.

        Returns:
            list: State vector and covariance array of the update.
        """
        estimate_kps3d_list = []
        for i in range(self.n_kps3d):
            estimate_kps3d, _ = self.kf[i].get_update()
            estimate_kps3d_list.append(estimate_kps3d[:3])
        return np.array(estimate_kps3d_list)

    def get_state(self) -> np.ndarray:
        """Returns the current keypoints3d estimate.

        Returns:
            np.ndarray: The current keypoints3d.
        """
        keypoints3d = []
        for i in range(self.n_kps3d):
            keypoints3d.append(self.kf[i].x[:3])

        return np.array(keypoints3d).squeeze()
