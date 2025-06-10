import numpy as np
import matplotlib.pyplot as plt


k = 0.5  # control gain
Kp = 1.0  # speed proportional gain
dt = 0.1  # [s] time difference
L = 2.9  # [m] Wheel base of vehicle

Lr = L / 2.  # [m]
Lf = L - Lr  # [m]
Cf = 32.  # [N/rad]
Cr = 34.  # [N/rad]
Iz = 2250.  # [kg/m2]
m = 1500.  # [kg]

ks=5



class State(object):
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, acceleration, delta):
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / L * np.tan(delta) * dt
        self.yaw = normalize_angle(self.yaw)
        self.v += acceleration * dt


class DynamicBicycleState(object):
    def __init__(self, x=0.0, y=0.0, yaw=0.0, vx=0.01, vy=0, omega=0.0,
                 max_vx=30.0, max_vy=30.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.omega = omega
        self.c_a = 1.36
        self.c_r1 = 0.01

        self.max_vx = max_vx
        self.max_vy = max_vy

    def update(self, throttle, delta):
        self.x = self.x + self.vx * np.cos(self.yaw) * dt - self.vy * np.sin(self.yaw) * dt
        self.y = self.y + self.vx * np.sin(self.yaw) * dt + self.vy * np.cos(self.yaw) * dt
        self.yaw = self.yaw + self.omega * dt
        self.yaw = normalize_angle(self.yaw)
        Ffy = -Cf * np.arctan2(((self.vy + Lf * self.omega) / self.vx - delta), 1.0)
        Fry = -Cr * np.arctan2((self.vy - Lr * self.omega) / self.vx, 1.0)
        R_x = self.c_r1 * self.vx
        F_aero = self.c_a * self.vx ** 2
        F_load = F_aero + R_x



        self.vx = self.vx + (throttle - Ffy * np.sin(delta) / m - F_load / m + self.vy * self.omega) * dt
        self.vy = self.vy + (Fry / m + Ffy * np.cos(delta) / m - self.vx * self.omega) * dt
        self.omega = self.omega + (Ffy * Lf * np.cos(delta) - Fry * Lr) / Iz * dt

        if self.vx > self.max_vx:
            self.vx = self.max_vx
        elif self.vx < -self.max_vx:
            self.vx = -self.max_vx
        if self.vy > self.max_vy:
            self.vy = self.max_vy
        elif self.vy < -self.max_vy:
            self.vy = -self.max_vy

    def initialize_rendering(self):
        plt.ion()
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('key_release_event',
                                    lambda event: [exit(0) if event.key == 'escape' else None])

        self.ax = self.fig.add_subplot(111, projection='3d')

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle

