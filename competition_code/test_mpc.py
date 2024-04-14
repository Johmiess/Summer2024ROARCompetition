import unittest
from mpc import MPCController
import numpy as np

class TestMPCController(unittest.TestCase):
    def setUp(self):
        # Create an instance of MPCController for testing
        self.controller = MPCController()

    def test_acceleration_from_throttle_and_speed(self):
        # Test the acceleration_from_throttle_and_speed method
        throttle = 0.5
        speed = 20
        expected_acceleration = throttle * self.controller.max_acceleration * (1 - self.controller.acc_speed_intercept * speed)
        actual_acceleration = self.controller.acceleration_from_throttle_and_speed(throttle, speed)
        self.assertAlmostEqual(actual_acceleration, expected_acceleration)

    def test_compute_errors(self):
        # Test the compute_errors method
        state = [0, 0, 0, 30]
        expected_cte = 0  # Since the vehicle is on the reference trajectory
        expected_epsi = 0  # Since the vehicle is aligned with the reference trajectory
        actual_cte, actual_epsi = self.controller.compute_errors(state)
        self.assertAlmostEqual(actual_cte, expected_cte)
        self.assertAlmostEqual(actual_epsi, expected_epsi)

    def test_update_state(self):
        # Test the update_state method
        state = [0, 0, 0, 30]
        control_input = [0.1, 0.5]
        expected_state = [0, 0, 0, 30]
        expected_state[0] += expected_state[3] * np.cos(expected_state[2]) * self.controller.dt
        expected_state[1] += expected_state[3] * np.sin(expected_state[2]) * self.controller.dt
        expected_state[2] += expected_state[3] / self.controller.length * np.tan(control_input[0]*self.controller.max_steering) * self.controller.dt
        expected_state[3] += self.controller.acceleration_from_throttle_and_speed(control_input[1], expected_state[3]) * self.controller.dt
        expected_state[2] = self.controller.normalize_angle(expected_state[2])
        actual_state = self.controller.update_state(state, control_input)

        for i in range(len(expected_state)):
            self.assertAlmostEqual(actual_state[i], expected_state[i])

    def test_normalize_angle(self):
        # Test the normalize_angle method
        angle = np.pi + 0.5
        expected_normalized_angle = -np.pi + 0.5
        actual_normalized_angle = self.controller.normalize_angle(angle)
        self.assertAlmostEqual(actual_normalized_angle, expected_normalized_angle)

    def test_solve_mpc(self):
        # Test the solve_mpc method
        initial_state = [0, 0, 0, 30]
        expected_optimal_control_input = [0,0]  # Since the initial state is on the reference trajectory
        actual_optimal_control_input = self.controller.solve_mpc(initial_state)
        self.assertAlmostEqual(actual_optimal_control_input, expected_optimal_control_input)

if __name__ == '__main__':
    unittest.main()