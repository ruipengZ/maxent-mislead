import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation

from matplotlib.patches import Rectangle, Ellipse, Polygon


dt = 1  # [s] time difference
max_simulation_time = 50
max_vel = 3

dest_x = 3
dest_y = 0

wall_start_x = 2
wall_width = 0.05

wall_height = 2.


class State(object):

    def __init__(self, x=0.0, y=0.0):
        super(State, self).__init__()
        self.x = x
        self.y = y

    def update(self, v_x, v_y):
        vel_x = np.clip(v_x, -max_vel, max_vel)
        vel_y = np.clip(v_y, -max_vel, max_vel)
        self.x += vel_x * dt
        self.y += vel_y * dt

    def simulate_step(self, v_x, v_y):
        vel_x = np.clip(v_x, -max_vel, max_vel)
        vel_y = np.clip(v_y, -max_vel, max_vel)
        next_x = self.x + vel_x * dt
        next_y = self.y + vel_y * dt
        return next_x, next_y




class Obstacle2D_Env(gym.Env):
    def __init__(self, render_mode=None, **kwargs):
        super(Obstacle2D_Env, self).__init__()
        vel_high = np.array([max_vel, max_vel], dtype=np.float32)

        spc_xl = -20
        spc_xr = 20
        spc_yb = -10
        spc_yt = 10

        self.state_x_mean = 10
        self.state_x_range = 20
        self.state_y_mean = 0
        self.state_y_range = 20
        self.obstacles_size_range = 20


        obs_space = np.array(
            [
                [spc_xl, spc_xr],
                [spc_yb, spc_yt],
            ]
            , dtype=np.float32
        )
        obs_low = obs_space[:, 0]
        obs_high = obs_space[:, 1]

        self.action_space = spaces.Box(
            low=-vel_high,
            high=vel_high,
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,

        )

        self.seed()

        self.obstacles = [

            [(wall_start_x, wall_height), (wall_start_x, -wall_height), (wall_start_x+wall_width, wall_height), (wall_start_x+wall_width, -wall_height)],
        ]

        self.obstacles_width = np.array([abs(obs[2][0] - obs[0][0]) for obs in self.obstacles], dtype=np.float32)
        self.obstacles_height = np.array([abs(obs[0][1] - obs[1][1]) for obs in self.obstacles], dtype=np.float32)
        self.obstacles_center_x = np.array([(obs[2][0] + obs[0][0]) / 2 for obs in self.obstacles], dtype=np.float32)
        self.obstacles_center_y = np.array([(obs[0][1] + obs[1][1]) / 2 for obs in self.obstacles], dtype=np.float32)


        self.dest_x = dest_x
        self.dest_y = dest_y

        self.fig = None
        self.movie_fig = plt.figure()
        self.movie_ax = self.movie_fig.add_subplot(111)

        self.plot_first_called = True

    @staticmethod
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


    def intersect(self, A, B, C, D):
        return (self.ccw(A, C, D) != self.ccw(B, C, D)) and (self.ccw(A, B, C) != self.ccw(A, B, D))


    @staticmethod
    def point_inside_rect(p, rect):
        x, y = p
        x1, y1, x2, y2 = rect
        return x1 < x < x2 and y1 < y < y2

    def line_rect_collision(self, line, rect):
        x1, y1, x2, y2 = line
        rx1, ry1, rx2, ry2 = rect

        if self.point_inside_rect((x1, y1), rect) or self.point_inside_rect((x2, y2), rect):
            return True

        return (self.intersect((x1, y1), (x2, y2), (rx1, ry1), (rx1, ry2)) or  # left
                self.intersect((x1, y1), (x2, y2), (rx1, ry1), (rx2, ry1)) or  # top
                self.intersect((x1, y1), (x2, y2), (rx2, ry1), (rx2, ry2)) or  # right
                self.intersect((x1, y1), (x2, y2), (rx1, ry2), (rx2, ry2)))  # bottom


    def check_collision(self, next_x, next_y):
        hit = False

        cur_x, cur_y = self.state.x, self.state.y
        for obstacle in self.obstacles:
            wall_top_left = obstacle[0]
            wall_bottom_right = obstacle[-1]
            rect = wall_top_left + wall_bottom_right

            hit = self.line_rect_collision((cur_x, cur_y, next_x, next_y), rect)

            if hit:
                break

        return hit


    def check_goal(self, x, y):
        return self.dist_to_goal(x, y) < 0.1


    def dist_to_goal(self, x, y):
        return np.sqrt((x - self.dest_x) ** 2 + (y - self.dest_y) ** 2)


    def _get_reward(self, action, truncated):
        collision = False
        goal = False

        if truncated:
            return 0.0, collision, goal
        else:
            next_x, next_y = self.state.simulate_step(action[0], action[1])
            if self.check_collision(next_x, next_y):
                collision = True
                return -200.0, collision, goal
            elif self.check_goal(next_x, next_y):
                goal = True
                return 500.0, collision, goal

            else:
                reward = self.dist_to_goal(self.state.x, self.state.y) - self.dist_to_goal(next_x, next_y)
                return reward, collision, goal



    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        return [seed]



    def render(self):
        if self.plot_first_called:
            self._ini_render()
            self.plot_first_called = False
        plt.cla()
        self._plot_init(self.ax)


        self.ax.set_title(
            f'Time: {self.time:.1f} s, Position: ({self.xs[-1]:.1f} , {self.ys[-1]:.1f})')
        self.ax.plot(self.xs[-1], self.ys[-1], 'o', c='b', markersize=5)
        self.ax.plot(self.xs, self.ys, 'b-')

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


        return self.fig

    def _plot_init(self, ax):

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, which='both', linestyle=':', linewidth=1)
        for obstacle in self.obstacles:
            ax.add_patch(Rectangle((obstacle[1][0], obstacle[1][1]), obstacle[2][0] - obstacle[0][0],
                                   obstacle[0][1] - obstacle[1][1], facecolor='grey', edgecolor='white', linestyle='-',
                                   linewidth=1.5, hatch='//'))



        dest_dot = patches.Circle((self.dest_x, self.dest_y), 0.05, facecolor='white', edgecolor='red', linewidth=3,)
        ax.add_patch(dest_dot)

        init_dot = patches.Circle((self.xs[0], self.ys[0]), 0.05, facecolor='white', edgecolor='green', linewidth=3,)
        ax.add_patch(init_dot)


    def _ini_render(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)




    def reset(self, **kwargs):

        self.state = State(x=0., y=0.)


        self.time = 0.0

        self.xs = [self.state.x]
        self.ys = [self.state.y]
        self.ts = [0.0]


        info = {'state':(self.state.x, self.state.y)}
        self.plot_first_called = True
        return self._get_obs(), info


    def _get_obs(self):
        obs = np.array([(self.state.x - (wall_start_x + wall_width/2)) / (wall_start_x + wall_width/2),
                        self.state.y / 3,
                        ], dtype=np.float32)


        return obs



    def step(self, action):
        # action: a(x), a(y)
        vx = action[0]
        vy = action[1]


        terminal = False
        truncated = False

        self.time += dt
        if self.time > max_simulation_time:
            truncated = True

        reward, collision, goal = self._get_reward(action, truncated)  # simulate step inside
        if collision or goal:
            terminal = True

        self.state.update(vx, vy)

        self.xs.append(self.state.x)
        self.ys.append(self.state.y)
        self.ts.append(self.time)



        info = {'state':(self.state.x, self.state.y)}

        return self._get_obs(), reward, terminal, truncated, info


if __name__ == '__main__':
    env = Obstacle2D_Env()
    env.reset()
    for i in range(100):
        obs, rew, terminal, truncated, info = env.step([0.1,0])
        env.render()
        print(env.state.x, env.state.y, rew)
        if truncated or terminal:
            break
    env.close()