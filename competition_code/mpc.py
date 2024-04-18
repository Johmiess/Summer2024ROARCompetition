import numpy as np
from scipy.optimize import minimize
from shapely.geometry import LineString, Point
import bisect

class MPCController:
    def __init__(self, dt=0.05, horizon=10, reference_trajectory=None, log=False):
        self.dt = dt  # Time step for MPC
        self.horizon = horizon  # MPC horizon
        self.length = 3.0  # Example length of the vehicle
        self.target_speed = 30  # Reference speed
        self.wt1 = 100  # Weight for cte
        self.wt2 = 100  # Weight for epsi
        self.wt3 = 1  # Weight for speed error
        self.wt4 = 1  # Weight for actuations
        self.wt5 = 10  # Weight for actuation rate of change
        self.reference_trajectory = reference_trajectory if reference_trajectory is not None else np.zeros((horizon, 2))
        self.ref_line = LineString(self.reference_trajectory[:, :2])

        self.max_acceleration = 4.0  # Maximum acceleration
        self.acc_speed_intercept = -0.04

        self.max_steering = 70 * np.pi / 180  # Maximum steering angle

        self.last_predicted_actuation = np.array([0,0.8] * self.horizon)

        self.log = log
    
    def acceleration_from_throttle_and_speed(self, throttle, speed):
        return throttle * self.max_acceleration * (1 - self.acc_speed_intercept * speed)

    def mpc_cost_function(self, initial_state, control_inputs):
        assert len(control_inputs) == self.horizon * 2, f"Control inputs shape {control_inputs.shape} does not match horizon {self.horizon}"
        # Initialize cost
        total_cost = 0.0
        state = initial_state.copy()
        
        # Calculate cost for each time step
        for i in range(self.horizon):
            delta, th = control_inputs[2*i], control_inputs[2*i+1]
            # Compute cross track error (cte) and orientation error (epsi)
            cte, epsi = self.compute_errors(state)
            
            speed_error = state[3] - self.target_speed  # Speed error
            
            # Compute cost for each term
            cost_cte = self.wt1 * cte**2
            cost_epsi = self.wt2 * epsi**2
            cost_speed = self.wt3 * speed_error**2
            cost_actuations = self.wt4 * (th**2 + delta**2)
            
            if i > 0:
                prev_delta, prev_th = control_inputs[2*(i-1)], control_inputs[2*(i-1)+1]
                cost_steering_rate = self.wt5 * ((delta - prev_delta) / self.dt)**2
                cost_throttle_rate = self.wt5 * ((th - prev_th) / self.dt)**2
            else:
                cost_steering_rate = 0
                cost_throttle_rate = 0
            
            # Total cost for this time step
            total_cost += cost_cte + cost_epsi + cost_speed + cost_actuations + cost_steering_rate + cost_throttle_rate
            # Update state using bicycle model
            state = self.update_state(state, [delta, th])
            
        return total_cost

    def compute_errors(self, state):
        # Find the closest point on the reference trajectory to the vehicle
        x, y = state[0], state[1]
        point = Point(x, y)
        proj = self.ref_line.project(point)
        closest_point = self.ref_line.interpolate(proj)
        cte = np.linalg.norm([x - closest_point.x, y - closest_point.y])

        # Compute orientation error by finding the angle between the vehicle orientation and the trajectory
        orientation = state[2]
        next_point = self.ref_line.interpolate(proj + 0.1)
        ref_orientation = np.arctan2(next_point.y - closest_point.y, next_point.x - closest_point.x)
        epsi = self.normalize_angle(orientation - ref_orientation)

        return cte, epsi
    
    def update_state(self, state, control_input):
        # Update state using bicycle model
        delta = -control_input[0]*self.max_steering  # Steering angle
        th = control_input[1]  # Throttle/Brake
        a = self.acceleration_from_throttle_and_speed(th, state[3])  # Acceleration
        
        state[0] += state[3] * np.cos(state[2]) * self.dt
        state[1] += state[3] * np.sin(state[2]) * self.dt
        state[2] += state[3] / self.length * np.tan(delta) * self.dt
        state[3] += a * self.dt
        
        # Normalize orientation
        state[2] = self.normalize_angle(state[2])

        self.prediction_time += self.dt

        return state
    
    def normalize_angle(self, angle):
        # Normalize angle between -pi and pi
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def solve_mpc(self, initial_state, current_time=None):
        self.current_time = current_time
        self.prediction_time = current_time 
        # Initial guess for control inputs using the last predicted actuation
        initial_guess = self.last_predicted_actuation
        #bounds are -1 to 1 for both steering and throttle
        bounds = np.array([(-1, 1)] * self.horizon * 2)
        assert len(initial_guess) == len(bounds), f"Initial guess shape {initial_guess.shape} does not match bounds shape {bounds.shape}"
        result = minimize(lambda x: self.mpc_cost_function(initial_state, x), initial_guess, bounds=bounds)
        optimal_control_inputs = result.x

        if self.log:
            with open("control_log.csv", "a") as f:
                for i in range(len(optimal_control_inputs)//2):
                    f.write(f"{self.current_time}, {self.current_time+(i+1)*self.dt}, {optimal_control_inputs[2*i]}, {optimal_control_inputs[2*i+1]}\n")

        #return only the first control input
        self.last_predicted_actuation = optimal_control_inputs
        return [optimal_control_inputs[0], optimal_control_inputs[1]]

class MPCShadowController:
    def __init__(self, dt=0.05, horizon=10, reference_line=None, shadow=None, start_time=0, log=False):
        self.dt = dt  # Time step for MPC
        self.horizon = horizon  # MPC horizon
        self.length = 3.0  # Example length of the vehicle
        self.wt1 = 100  # Weight for cte
        self.wt2 = 100  # Weight for epsi
        self.wt3 = 100  # Weight for diff with shadow progress error
        self.wt4 = 1  # Weight for actuations
        self.wt5 = 10  # Weight for actuation rate of change
        self.reference_line = reference_line
        self.ref_line = LineString(reference_line)
        self.shadow = shadow # 2D array with [time, progress] for shadow trajectory, time should start from 0
        self.start_time = start_time
        self.lap = 1
        self.n_laps = 3
        self.prev_progress = 0

        self.max_acceleration = 4.0  # Maximum acceleration
        self.acc_speed_intercept = -0.04

        self.max_steering = 70 * np.pi / 180  # Maximum steering angle

        self.last_predicted_actuation = np.array([0,0.8] * self.horizon)

        self.log = log
    
    def acceleration_from_throttle_and_speed(self, throttle, speed):
        return throttle * self.max_acceleration * (1 - self.acc_speed_intercept * speed)
    
    def compute_shadow_progress_error(self, state):
        # Find the closest point on the reference trajectory to the vehicle
        x, y = state[0], state[1]
        point = Point(x, y)
        progress_lap = self.ref_line.project(point, normalized=True)
        if progress_lap <0.1 and self.prev_progress > 0.9:
            self.lap += 1
        self.prev_progress = progress_lap
        progress = (progress_lap + self.lap - 1) / self.n_laps

        # Find idx of the shadow trajectory that is closest to the current time
        elapsed_time = self.prediction_time - self.start_time + 10 # Add 10 seconds ahead as objective is to be ahead of the shadow trajectory
        idx = bisect.bisect_left(self.shadow[:, 0], elapsed_time)
        shadow_progress = self.shadow[idx][1]
        target_progress = shadow_progress * 1.01  # 1% ahead of the shadow trajectory

        return progress - target_progress
        

    def mpc_cost_function(self, initial_state, control_inputs):
        assert len(control_inputs) == self.horizon * 2, f"Control inputs shape {control_inputs.shape} does not match horizon {self.horizon}"
        # Initialize cost
        total_cost = 0.0
        state = initial_state.copy()
        
        # Calculate cost for each time step
        for i in range(self.horizon):
            delta, th = control_inputs[2*i], control_inputs[2*i+1]
            # Compute cross track error (cte) and orientation error (epsi)
            cte, epsi = self.compute_errors(state)
            
            # Compute shadow progress error
            shadow_progress_error = self.compute_shadow_progress_error(state)
            
            # Compute cost for each term
            cost_cte = self.wt1 * cte**2
            cost_epsi = self.wt2 * epsi**2
            cost_shadow = self.wt3 * shadow_progress_error**2
            cost_actuations = self.wt4 * (th**2 + delta**2)
            
            if i > 0:
                prev_delta, prev_th = control_inputs[2*(i-1)], control_inputs[2*(i-1)+1]
                cost_steering_rate = self.wt5 * ((delta - prev_delta) / self.dt)**2
                cost_throttle_rate = self.wt5 * ((th - prev_th) / self.dt)**2
            else:
                cost_steering_rate = 0
                cost_throttle_rate = 0
            
            # Total cost for this time step
            total_cost += cost_cte + cost_epsi + cost_shadow + cost_actuations + cost_steering_rate + cost_throttle_rate
            # Update state using bicycle model
            state = self.update_state(state, [delta, th])
            
        return total_cost

    def compute_errors(self, state):
        # Find the closest point on the reference trajectory to the vehicle
        x, y = state[0], state[1]
        point = Point(x, y)
        proj = self.ref_line.project(point)
        closest_point = self.ref_line.interpolate(proj)
        cte = np.linalg.norm([x - closest_point.x, y - closest_point.y])

        # Compute orientation error by finding the angle between the vehicle orientation and the trajectory
        orientation = state[2]
        next_point = self.ref_line.interpolate(proj + 0.1)
        ref_orientation = np.arctan2(next_point.y - closest_point.y, next_point.x - closest_point.x)
        epsi = self.normalize_angle(orientation - ref_orientation)

        return cte, epsi
    
    def update_state(self, state, control_input):
        # Update state using bicycle model
        delta = -control_input[0]*self.max_steering  # Steering angle
        th = control_input[1]  # Throttle/Brake
        a = self.acceleration_from_throttle_and_speed(th, state[3])  # Acceleration
        
        state[0] += state[3] * np.cos(state[2]) * self.dt
        state[1] += state[3] * np.sin(state[2]) * self.dt
        state[2] += state[3] / self.length * np.tan(delta) * self.dt
        state[3] += a * self.dt
        
        # Normalize orientation
        state[2] = self.normalize_angle(state[2])

        self.prediction_time += self.dt

        return state
    
    def normalize_angle(self, angle):
        # Normalize angle between -pi and pi
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def solve_mpc(self, initial_state, current_time=None):
        self.current_time = current_time
        self.prediction_time = current_time 
        # Initial guess for control inputs using the last predicted actuation
        initial_guess = self.last_predicted_actuation
        #bounds are -1 to 1 for both steering and throttle
        bounds = np.array([(-1, 1)] * self.horizon * 2)
        assert len(initial_guess) == len(bounds), f"Initial guess shape {initial_guess.shape} does not match bounds shape {bounds.shape}"
        result = minimize(lambda x: self.mpc_cost_function(initial_state, x), initial_guess, bounds=bounds)
        optimal_control_inputs = result.x

        if self.log:
            with open("control_log.csv", "a") as f:
                for i in range(len(optimal_control_inputs)//2):
                    f.write(f"{self.current_time}, {self.current_time+(i+1)*self.dt}, {optimal_control_inputs[2*i]}, {optimal_control_inputs[2*i+1]}\n")

        #return only the first control input
        self.last_predicted_actuation = optimal_control_inputs
        return [optimal_control_inputs[0], optimal_control_inputs[1]]