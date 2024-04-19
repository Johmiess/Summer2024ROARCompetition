import numpy as np

class VehicleDynamics:
    def __init__(self):
        # Hardcoded parameters from the provided data
        self.params = {
            'torque_curve': [
                {'rpm': 0.0, 'torque': 400.0},
                {'rpm': 1890.760742, 'torque': 500.0},
                {'rpm': 4800.465332, 'torque': 499.610199}
            ],
            'max_rpm': 15000.0,
            'moi': 1.0,
            'mass': 1845.0,
            'final_ratio': 4.0,
            'center_of_mass_x': 0.45,
            'damping_rate_full_throttle': 0.15,
            'damping_rate_zero_throttle_clutch_engaged': 2.0,
            'damping_rate_zero_throttle_clutch_disengaged': 0.35,
            'wheels': [
                {
                    'lat_stiff_value': 20.0,
                    'lat_stiff_max_load': 3.0,
                    'tire_friction': 3.5,
                    'radius': 37.0,
                    'max_brake_torque': 1500.0
                },
                {
                    'lat_stiff_value': 20.0,
                    'lat_stiff_max_load': 3.0,
                    'tire_friction': 3.5,
                    'radius': 37.0,
                    'max_brake_torque': 1500.0
                }
            ]
        }
        
        self.vel = np.array([0.0, 0.0])  # velocity [v_x, v_y]
        self.acc = np.array([0.0, 0.0])  # acceleration [a_x, a_y]
        self.pos = np.array([0.0, 0.0])  # position [x, y]
        self.yaw_rate = 0.0
        self.yaw = 0.0

    def update(self, throttle, brake, steering_angle, dt):
        # Calculate engine torque based on RPM
        rpm = np.linalg.norm(self.vel) * self.params['final_ratio'] * 60 / (2 * np.pi * self.params['wheels'][0]['radius'])
        engine_torque = self.get_engine_torque(rpm)
        
        # Calculate longitudinal force from throttle and brake
        throttle_torque = throttle * engine_torque
        brake_torque = brake * self.params['wheels'][0]['max_brake_torque']
        total_torque = throttle_torque - brake_torque
        
        # Update acceleration and velocity
        self.acc[0] = total_torque / self.params['mass']
        self.acc[1] = 0.0  # Assuming no vertical acceleration
        self.vel += self.acc * dt
        
        # Update position
        self.pos += self.vel * dt
        
        # Calculate lateral force from tire slip
        lateral_force = self.get_lateral_force(steering_angle, np.linalg.norm(self.vel))
        
        # Update yaw rate and yaw
        self.yaw_rate = lateral_force * self.params['center_of_mass_x'] / self.params['moi']
        self.yaw += self.yaw_rate * dt

        # Update position based on yaw (heading)
        self.pos[0] += self.vel[0] * np.cos(self.yaw) * dt - self.vel[1] * np.sin(self.yaw) * dt
        self.pos[1] += self.vel[0] * np.sin(self.yaw) * dt + self.vel[1] * np.cos(self.yaw) * dt

    def get_engine_torque(self, rpm):
        # Interpolate torque from torque curve
        rpm_values = [point['rpm'] for point in self.params['torque_curve']]
        torque_values = [point['torque'] for point in self.params['torque_curve']]
        
        return np.interp(rpm, rpm_values, torque_values, left=0, right=0)

    def get_lateral_force(self, steering_angle, vel_magnitude):
        # Calculate slip angle
        slip_angle = np.arctan2(self.yaw_rate * self.params['center_of_mass_x'] + vel_magnitude * np.sin(steering_angle), vel_magnitude * np.cos(steering_angle))
        
        # Pacejka Magic Formula parameters
        B = self.params['wheels'][0]['lat_stiff_value']
        C = self.params['wheels'][0]['lat_stiff_max_load'] / self.params['mass']
        D = self.params['wheels'][0]['tire_friction'] * self.params['mass'] * 9.81
        E = 0.97  # Empirical value
        
        # Calculate lateral force using Pacejka Magic Formula
        lateral_force = D * np.sin(C * np.arctan(B * slip_angle - E * (B * slip_angle - np.arctan(B * slip_angle))))
        
        return lateral_force

    def predict(self, initial_state, control_inputs, dt):
        self.pos = initial_state[:2]
        self.yaw = initial_state[2]
        self.vel = initial_state[3]

        preds = []
        
        for i in range(len(control_inputs) // 2):
            th, st = control_inputs[2*i], control_inputs[2*i+1]
            br = 0.0 if th > 0 else -th
            th = max(0, th)
            self.update(th, br, st, dt)
            preds.append([self.pos[0], self.pos[1], self.yaw, self.vel[0], self.vel[1], self.acc[0], self.acc[1], self.yaw_rate])
        
        return preds