import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import os
import numpy as np
import matplotlib.pyplot as plt
from .base_quadrotor import Quadrotor

# Simulation parameters
g = 9.81
m = 0.65
jx = 7.5e-1
jy = 7.5e-1
jz = 1.3
kt = 31.3e-2
kd = 7.5e-4
L = 0.23
T = 4
dt = 0.1


class QuadRotorEnv(gym.Env):
    
    def __init__(self):
        
        self.x_pos = 0
        self.y_pos = 0
        self.z_pos = 0
        self.x_vel = 0
        self.y_vel = 0
        self.z_vel = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.roll_vel = 0
        self.pitch_vel = 0
        self.yaw_vel = 0

        self.dt = dt
        
        self.q = Quadrotor(x=self.x_pos, y=self.y_pos, z=self.z_pos, \
                           roll=self.roll, pitch=self.pitch, yaw=self.yaw, \
                           size=1
                           )


        low = np.full(9, -float('inf'), dtype=np.float64)
        high = np.full(9, float('inf'), dtype=np.float64)

        low_a = np.full(4, 0, dtype=np.float64)
        high_a = np.full(4, 10, dtype=np.float64)
        
        self.observation_space= spaces.Box(low, high, dtype=np.float64)
        self.action_space= spaces.Box(low_a, high_a, dtype=np.float64)
        self.seed()

        self.plot_first_called = True
        self.q.reset_trajectory_plot()



    def step(self, action):
        action = np.clip(action, 0, 10)

        w1 = action[0]
        w2 = action[1]
        w3 = action[2]
        w4 = action[3]
        # compute the total thrust and the moments of each rotational axis
        u1 = kt*(w1**2 + w2**2 + w3**2 + w4**2) # Thrust
        u2 = kt*(w1**2 - w3**2)# M1
        u3 = kt*(w2**2 - w4**2)# M2
        u4 = kd*(w1**2 + w3**2 - w2**2 - w4**2)# M3

        self.x_vel += ((np.cos(self.roll)*np.sin(self.pitch)*np.cos(self.yaw)+np.sin(self.roll)*np.sin(self.yaw))*u1/m)*self.dt
        self.y_vel += ((np.cos(self.roll)*np.sin(self.pitch)*np.sin(self.yaw)-np.sin(self.roll)*np.cos(self.yaw))*u1/m)*self.dt
        self.z_vel += (-g+(np.cos(self.roll)*np.cos(self.pitch))*u1/m)* self.dt

        self.x_pos += self.x_vel * self.dt
        self.y_pos += self.y_vel * self.dt
        self.z_pos += self.z_vel * self.dt

        roll_torque = self.yaw_vel*self.pitch_vel*(jy-jz)/ jx + L*u2/jx
        pitch_torque = self.yaw_vel*self.roll_vel*(jz-jx)/ jy + L*u3/jy
        yaw_torque = self.pitch_vel*self.roll_vel*(jx-jy)/ jz + L*u4/jz

        self.roll_vel += roll_torque * self.dt
        self.pitch_vel += pitch_torque * self.dt
        self.yaw_vel += yaw_torque * self.dt

        self.roll += self.roll_vel * self.dt
        self.pitch += self.pitch_vel * self.dt
        self.yaw += self.yaw_vel * self.dt
        
        self.q.update_pose(self.x_pos, self.y_pos, self.z_pos, \
                self.roll, self.pitch, self.yaw)

        self.t += self.dt

        if self.t< T:
            done = False
        else:
            done = True

        reward = self._get_reward(done)


        return self._get_obs(), reward, done, False, {}

    def seed(self, seed=None):

        self.np_random, seed = seeding.np_random(seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        return [seed]
    
    def reset(self, **kwargs):

        self.t = 0

        self.x_pos = self.np_random.uniform(-0.1, 0.1)
        self.y_pos = self.np_random.uniform(-0.1, 0.1)
        self.z_pos = self.np_random.uniform(-1, 1)

        vbound = 0.1

        self.x_vel = self.np_random.uniform(-vbound, vbound)
        self.y_vel = self.np_random.uniform(-vbound, vbound)
        self.z_vel = self.np_random.uniform(-vbound, vbound)

        self.roll = 0 
        self.pitch = 0 
        self.yaw = 0 
        self.roll_vel = 0
        self.pitch_vel = 0
        self.yaw_vel = 0


        self.traj()

        self.q.update_pose(self.x_pos, self.y_pos, self.z_pos, \
                self.roll, self.pitch, self.yaw)


        info = {}
        return self._get_obs(), info
   
    def set_state(self, state, time):
        self.x_pos = state[0]
        self.y_pos = state[1]
        self.z_pos = state[2]
        self.roll = state[3]
        self.pitch = state[4]
        self.yaw = state[5]
        self.x_vel = state[6]
        self.y_vel = state[7]
        self.z_vel = state[8]
        self.roll_vel = state[9]
        self.pitch_vel = state[10]
        self.yaw_vel = state[11]
        self.t = time
        self.traj()


        return self._get_obs()


    def render(self, mode = 'human', close = True, color = 'black', clear = True):
        if self.plot_first_called:
            self.q.initialize_rendering()
            self.plot_first_called = False
    
        self.q.plot(color=color, clear=clear)


        # Redraw
        self.q.fig.canvas.draw()
        self.q.fig.canvas.flush_events()
        plt.pause(0.001)

        return self.q.ax



    def _get_obs(self):
        obs = np.array([(self.des_x_pos-self.x_pos), (self.des_y_pos-self.y_pos), (self.des_z_pos-self.z_pos),\
                        self.roll, self.pitch, self.yaw, (self.des_x_vel - self.x_vel),\
                        (self.des_y_vel - self.y_vel), (self.des_z_vel - self.z_vel)])
        self.traj()
        return obs
   
    
    def _get_reward(self, done):

        r = 5*(-abs(self.des_x_pos-self.x_pos)-abs(self.des_y_pos-self.y_pos)-abs(self.des_z_pos-self.z_pos))\
        -abs(self.yaw)-abs(self.roll)-abs(self.pitch)

        return r


    def traj(self):
        
        self.des_x_pos = 1.5*self.t
        self.des_y_pos = 0
        self.des_z_pos = 0
        self.des_x_vel = 1.5
        self.des_y_vel = 0
        self.des_z_vel = 0
        self.des_x_acc = 0
        self.des_y_acc = 0
        self.des_z_acc = 0


if __name__ == '__main__':
    env = QuadRotorEnv()
    env.reset()
    for i in range(100):
        obs, rew, terminal, truncated, info = env.step([0.5,0.5,0.5,0.5])
        env.render()
        # print(env.state.x, env.state.y, rew)
        if truncated or terminal:
            break
    env.close()