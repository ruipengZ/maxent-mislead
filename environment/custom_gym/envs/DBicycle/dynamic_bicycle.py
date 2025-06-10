
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import numpy as np
import matplotlib.pyplot as plt

from environment.custom_gym.envs.DBicycle import cubic_spline_planner
from environment.custom_gym.envs.DBicycle.bicycle_controller import DynamicBicycleState, dt, L

target_speed = 6  # [m/s]
max_simulation_time = 20.0



class MovingTarget:
    def __init__(self, ax, ay, speed, start_distance=13.0):

        self.speed = speed
        self.dt = dt
        self.max_simulation_time = max_simulation_time
        self.time_steps = 110

        self.sp = cubic_spline_planner.CubicSpline2D(ax, ay)  # Spline for interpolation

        self.start_distance = start_distance

        self.current_idx = 0
        self.positions = []
        self.speeds =[]
        self.speed_profile = lambda t: min(6, 1 + 0.1 * t)

        self._precompute_positions()

    def _precompute_positions(self):
        current_s = self.start_distance
        time = 0.0
        for _ in range(self.time_steps):
            speed = self.speed_profile(time)
            self.speeds.append(speed)
            x, y = self.sp.calc_position(current_s)
            yaw = self.sp.calc_yaw(current_s)
            curvature = self.sp.calc_curvature(current_s)
            self.positions.append((x, y, yaw, curvature))
            current_s += speed * self.dt
            time += self.dt


    @property
    def x(self):
        return self.positions[self.current_idx][0] if self.current_idx < len(self.positions) else None

    @property
    def y(self):
        return self.positions[self.current_idx][1] if self.current_idx < len(self.positions) else None

    @property
    def yaw(self):
        return self.positions[self.current_idx][2] if self.current_idx < len(self.positions) else None

    @property
    def curvature(self):
        return self.positions[self.current_idx][3] if self.current_idx < len(self.positions) else None

    @property
    def vx(self):
        return self.speeds[self.current_idx] * np.cos(self.yaw)

    @property
    def vy(self):
        return self.speeds[self.current_idx] * np.sin(self.yaw)

    def reset(self):
        self.current_idx = 0


    def update(self):
        if self.current_idx < len(self.positions) - 1:
            self.current_idx += 1


    def get_position(self):
        return self.x, self.y, self.yaw, self.curvature



class DBicycle_Env(gym.Env):
    def __init__(self, max_throttle=1, max_steer=1, init_vx=3, record=False):
        self.max_steer = max_steer  # [rad] max steering angle
        self.max_throttle = max_throttle

        high = np.array([100, 100, np.pi, 30, 30, 5.], dtype=np.float64)
        ac_high = np.array([max_throttle, max_steer], dtype=np.float64)

        self.action_space = spaces.Box(
            low=-ac_high,
            high=ac_high,
            dtype=np.float64
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float64
        )

        circle_points = 8
        radius = 10  # Radius of the circle
        angles = np.linspace(0*np.pi, 2 * np.pi, circle_points, endpoint=False)  # Equal spacing around the circle

        angles = np.flip(angles)

        ax = np.append(radius * np.cos(angles), radius * np.cos(angles[0]))
        ay = np.append(radius * np.sin(angles), radius * np.sin(angles[0]))



        self.target_speed = target_speed

        self.cx, self.cy, self.cyaw, self.ck, self.s = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)
        self.cx = self.cx[80:300]
        self.cy = self.cy[80:300]
        self.cyaw = self.cyaw[80:300]
        self.ck = self.ck[80:300]


        self.target = MovingTarget(ax,ay, speed=self.target_speed)
        start_s = 9.
        self.init_yaw = self.target.sp.calc_yaw(start_s)
        self.init_x,self.init_y = self.target.sp.calc_position(start_s)


        self.init_vy = 0
        self.init_vx = init_vx
        self.init_omega = -1

        self.goal = [self.cx[-1], self.cy[-1]]
        self.close_dist = 1.
        self.last_idx = len(self.cx) -1
        self.plot_first_called = True

        self.seed()
        self.record = record


    def is_close_to_goal(self, point):
        return np.linalg.norm(np.array(point) - np.array(self.goal)) < self.close_dist

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        return [seed]


    def step(self, action):

        ti = action[0]
        ti = np.clip(ti, -self.max_throttle, self.max_throttle)

        di = action[1]
        di = np.clip(di, -self.max_steer, self.max_steer)

        self.state.update(ti, di)

        self.time += dt

        self.xs.append(self.state.x)
        self.ys.append(self.state.y)
        self.yaws.append(self.state.yaw)

        self.target.update()
        if max_simulation_time >= self.time:
            done = False
        
        else:
            done = True


        reward, off_track, distance_target = self._get_reward()

        info = {
            'state_x': self.state.x,
            'state_y': self.state.y,
            'state_yaw': self.state.yaw,
            'state_vx': self.state.vx,
            'state_vy': self.state.vy,
            'state_omega': self.state.omega,
            'target_idx': self.target.current_idx,

        }

        return self._get_obs(), reward, done, False, info



    def reset(self, **kwargs):
        self.time = 0.0
        self.state = DynamicBicycleState(x = self.init_x,
                                         y = self.init_y,
                                         yaw= self.init_yaw,
                                         vx = self.init_vx,
                                         vy = self.init_vy,
                                         omega = self.init_omega)

        self.xs = [self.state.x]
        self.ys = [self.state.y]
        self.yaws = [self.state.yaw]

        self.target.reset()

        return self._get_obs(), {}
    
    def render(self, mode= 'human'):
        if self.plot_first_called:
            self.initialize_rendering()
            self.plot_first_called = False
        plt.cla()

        self.plot(self.target.x, self.target.y)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)



    def _get_obs(self):
        fx = self.state.x + L * np.cos(self.state.yaw)
        fy = self.state.y + L * np.sin(self.state.yaw)

        to_return = np.array([
            self.target.x - fx,
            self.target.y - fy,
            self.state.yaw,
            self.state.vx,
            self.state.vy,
            self.state.omega,
        ])

        if np.any(np.isinf(to_return)):
            assert False
        return to_return

    def _get_reward(self):
        off_track_penalty = abs(self.state.x**2+self.state.y**2-100) * 0.3
        distance_target_penalty = np.hypot(self.target.x - self.state.x, self.target.y - self.state.y)

        reward = - off_track_penalty -  distance_target_penalty
        return reward, -off_track_penalty, -distance_target_penalty

    def initialize_rendering(self):
        plt.ion()
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('key_release_event',
                                    lambda event: [exit(0) if event.key == 'escape' else None])

        self.ax = self.fig.add_subplot(111)



    def plot(self, target_x, target_y, arrow_length=3):

        a1 = self.ax.plot(self.cx, self.cy, "-r", label="course")
        fx = self.xs
        fy = self.ys

        car_length = L
        car_width = 1.0
        car_x = fx[-1]
        car_y = fy[-1]
        car_yaw = self.yaws[-1]

        # Calculate rectangle corners
        corners_x = np.array([car_length / 2, car_length / 2, -car_length / 2, -car_length / 2])
        corners_y = np.array([car_width / 2, -car_width / 2, -car_width / 2, car_width / 2])
        rotation_matrix = np.array([[np.cos(car_yaw), -np.sin(car_yaw)],
                                    [np.sin(car_yaw), np.cos(car_yaw)]])
        rotated_corners = rotation_matrix @ np.array([corners_x, corners_y])
        car_corners_x = rotated_corners[0, :] + car_x
        car_corners_y = rotated_corners[1, :] + car_y

        self.ax.fill(car_corners_x, car_corners_y, color="blue", alpha=0.5, label="car")
        self.ax.plot(fx, fy, "-b", label="trajectory")

        a3 = self.ax.plot(target_x, target_y, "xg", label="target")
        self.ax.arrow(
            fx[-1],
            fy[-1],
            arrow_length * np.cos(self.yaws[-1]),
            arrow_length * np.sin(self.yaws[-1]),
            head_width=0.3,
            head_length=0.5,
            fc="gray",
            ec="gray",
            label="facing direction"
        )
        self.ax.axis("equal")
        self.ax.grid(True)
        plt.show()




if __name__ == '__main__':
    env = DBicycle_Env()
    env.reset()
    for i in range(100):
        obs, rew, terminal, truncated, info = env.step([0.5,0.5])
        env.render()
        # print(env.state.x, env.state.y, rew)
        if truncated or terminal:
            break
    env.close()