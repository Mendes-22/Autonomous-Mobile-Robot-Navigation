import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, dt=0.1):
        self.dt = dt
        
        # State Vector [x, y, theta]
        self.x = np.array([[0.0], [0.0], [0.0]])
        
        # Covariance Matrix P (Initial uncertainty)
        self.P = np.eye(3) * 0.1
        
        # Process Noise Q (Trust in the physical motion model)
        self.Q = np.diag([0.05, 0.05, 0.01])
        
        # Measurement Noise R (Trust in sensors - GPS/IMU)
        self.R = np.diag([0.005, 0.005, 0.001])
        
        # Measurement Matrix H (Direct mapping between sensor and state)
        self.H = np.eye(3)

    def predict(self, v, w):
        """
        Predicts the next state using the kinematic motion model.
        v: linear velocity
        w: angular velocity
        """
        theta = self.x[2, 0]
        dist = v * self.dt
        
        # Motion equations (Kinematics)
        self.x[0, 0] += dist * np.cos(theta)
        self.x[1, 0] += dist * np.sin(theta)
        self.x[2, 0] += w * self.dt
        
        # Calculate the Jacobian matrix F
        F = np.array([
            [1, 0, -dist * np.sin(theta)],
            [0, 1,  dist * np.cos(theta)],
            [0, 0, 1]
        ])
        
        # Update Covariance: P = F * P * F.T + Q
        self.P = F @ self.P @ F.T + self.Q
        
        return self.x

    def update(self, z):
        """
        Updates the state estimate using actual sensor measurements.
        z: real sensor measurement [x, y, theta]
        """
        # Innovation (Difference between real and predicted measurement)
        y = z - (self.H @ self.x)
        
        # Normalize angle difference to [-pi, pi]
        y[2, 0] = (y[2, 0] + np.pi) % (2 * np.pi) - np.pi
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman Gain (Weighting factor)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Correction of state and covariance
        self.x = self.x + (K @ y)
        self.P = (np.eye(3) - (K @ self.H)) @ self.P
        
        return self.x
