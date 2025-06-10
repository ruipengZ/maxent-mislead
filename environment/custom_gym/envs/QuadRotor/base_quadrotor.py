import numpy as np
import matplotlib.pyplot as plt

class Quadrotor():
    
    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, size=0.25):

        self.p1 = np.array([size / 2, 0, 0, 1]).T
        self.p2 = np.array([-size / 2, 0, 0, 1]).T
        self.p3 = np.array([0, size / 2, 0, 1]).T
        self.p4 = np.array([0, -size / 2, 0, 1]).T
        self.x_data = []
        self.y_data = []
        self.z_data = []

        self.update_pose(x, y, z, roll, pitch, yaw)

    def update_pose(self, x, y, z, roll, pitch, yaw):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.x_data.append(x)
        self.y_data.append(y)
        self.z_data.append(z)


    def transformation_matrix(self):
        x = self.x
        y = self.y
        z = self.z
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw
        return np.array(
            [[np.cos(yaw) * np.cos(pitch), -np.sin(yaw) * np.cos(roll) + np.cos(yaw) * np.sin(pitch) * np.sin(roll), np.sin(yaw) * np.sin(roll) + np.cos(yaw) * np.sin(pitch) * np.cos(roll), x],
             [np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(roll) + np.sin(yaw) * np.sin(pitch)
              * np.sin(roll), -np.cos(yaw) * np.sin(roll) + np.sin(yaw) * np.sin(pitch) * np.cos(roll), y],
             [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(yaw), z]
             ])

    def initialize_rendering(self):
        plt.ion()
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('key_release_event',
                                    lambda event: [exit(0) if event.key == 'escape' else None])

        self.ax = self.fig.add_subplot(111, projection='3d')

    def reset_trajectory_plot(self):
        self.x_data = []
        self.y_data = []
        self.z_data = []

    def plot(self, color='k', clear=True, no_drone=False):  # pragma: no cover
        T = self.transformation_matrix()

        p1_t = np.matmul(T, self.p1)
        p2_t = np.matmul(T, self.p2)
        p3_t = np.matmul(T, self.p3)
        p4_t = np.matmul(T, self.p4)


        if clear:
            plt.cla()

        # plot target path
        t = np.linspace(0, 4, 40)

        des_x_pos = 1.5 * t
        des_y_pos = np.zeros(40)
        des_z_pos = np.zeros(40)

        self.ax.plot(des_x_pos, des_y_pos, des_z_pos, 'r-.', label='desired trajectory')

        # plot drone
        if not no_drone:
            a1 = self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                         [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                         [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], 'k.')

            a2 = self.ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
                         [p1_t[2], p2_t[2]], 'r-')
            a3 = self.ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]],
                         [p3_t[2], p4_t[2]], 'r-')

        a4 = self.ax.plot(self.x_data, self.y_data, self.z_data, color=color, linewidth=3)

        self.ax.set_zlim(-10, 3)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(-1, 1)

