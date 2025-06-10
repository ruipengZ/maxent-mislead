"""classic Acrobot task"""
import numpy as np
from gymnasium.error import DependencyNotInstalled
from numpy import sin, cos, pi
from scipy import integrate

import gymnasium as gym
from gymnasium import core, spaces
from gymnasium.utils import seeding
import pygame
from pygame import gfxdraw

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

np.random.seed(1)
# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

class AcrobotContEnv(core.Env):
    """
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.
    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.
    .. note::
        The dynamics equations were missing some terms in the NIPS paper which
        are present in the book. R. Sutton confirmed in personal correspondence
        that the experimental results shown in the paper and the book were
        generated with the equations shown in the book.
        However, there is the option to run the domain with the paper equations
        by setting book_or_nips = 'nips'
    **REFERENCE:**
    .. seealso::
        R. Sutton: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding (NIPS 1996)
    .. seealso::
        R. Sutton and A. G. Barto:
        Reinforcement learning: An introduction.
        Cambridge: MIT press, 1998.
    .. warning::
        This version of the domain uses the Runge-Kutta method for integrating
        the system dynamics and is more realistic, but also considerably harder
        than the original version which employs Euler integration,
        see the AcrobotLegacy class.
    """


    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 15,
    }

    dt = .2

    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1  #: moments of inertia for both links

    SCREEN_DIM = 500


    MAX_VEL_1 = 4 * pi  # 4 * pi #6 * pi # 10 * pi #
    MAX_VEL_2 = 9 * pi  # 9 * pi #13 * pi # 21 * pi #

    MAX_TORQUE = 20  # 20 before x6 x10

    torque_noise_max = 0.



    action_arrow = None
    domain_fig = None

    def __init__(self):
        self.viewer = None
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2])
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)

        self.action_space = spaces.Box(
            low=-self.MAX_TORQUE, high=self.MAX_TORQUE, shape=(1,), dtype=np.float64
        )

        self.state = None
        self.SCREEN_DIM = 500
        self.render_mode = "rgb_array"
        self.screen= None
        self.clock = None
        self.seed()

    def seed(self, seed=None):

        self.np_random, seed = seeding.np_random(seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        return [seed]

    def reset(self, **kwargs):
        surf = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        surf.fill((255, 255, 255))
        self.surf = surf
        self.timesteps = 0
        theta1 = self.np_random.uniform(low=np.deg2rad(45), high=np.deg2rad(135))
        s = np.concatenate(([theta1], np.random.uniform(low=-0.1, high=0.1, size=(3,))))
        self.state = s.tolist()


        return self._get_ob(), {}



    def step(self, a):
        s = self.state
        self.timesteps += 1
        torque = np.clip(a, -self.MAX_TORQUE, self.MAX_TORQUE)[0]

        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)


        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])


        ns = ns[-1]
        ns = ns[:4]  # omit action

        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns

        costs = angle_normalize(ns[0] - pi/2) ** 2 + angle_normalize(ns[1]) ** 2 + 0.1 * ns[2] ** 2 + 0.1 * ns[3] ** 2

        return (self._get_ob(), -costs, False, False,{})

    def _get_ob(self):
        s = self.state

        return np.array([cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])


    def _dsdt(self, s_augmented, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        # ======================================================================== discrete
        a = s_augmented[-1]  # torque
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]

        # ======================================================================== continuous
        # Precompute trigonometric functions
        sin_theta2 = sin(theta2)
        cos_theta2 = cos(theta2)
        sin_theta1 = sin(theta1)
        sin_theta1_theta2 = sin(theta1 + theta2)

        # Inertia matrix
        d11 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos_theta2) + I1 + I2
        d12 = m2 * (lc2**2 + l1 * lc2 * cos_theta2) + I2
        d21 = d12
        d22 = m2 * lc2**2 + I2
        D = np.array([[d11, d12], [d21, d22]])

        # Coriolis and centrifugal forces
        c1 = -m2 * l1 * lc2 * sin_theta2 * (2 * dtheta1 * dtheta2 + dtheta2**2)
        c2 = m2 * l1 * lc2 * sin_theta2 * dtheta1**2
        C = np.array([c1, c2])

        # Gravity vector
        g1 = (m1 * lc1 + m2 * l1) * g * np.cos(theta1) + m2 * lc2 * g * np.cos(theta1 + theta2)
        g2 = m2 * lc2 * g * np.cos(theta1 + theta2)
        G = np.array([g1, g2])

        # Control input
        tau = np.array([0, a])

        # Solve for angular accelerations
        ddtheta = np.linalg.inv(D) @ (tau - C - G)

        # Return the state derivative

        return (dtheta1, dtheta2, ddtheta[0], ddtheta[1], 0.)

    #         return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.)

    def render(self):
        if self.render_mode is None:

            print("You are calling render method without specifying any render mode. ")
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.SCREEN_DIM, self.SCREEN_DIM)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        if self.clock is None:
            self.clock = pygame.time.Clock()


        surf = self.surf
        s = self.state

        bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
        scale = self.SCREEN_DIM / (bound * 2)
        offset = self.SCREEN_DIM / 2

        if s is None:
            return None

        p1 = [
            -self.LINK_LENGTH_1 * cos(s[0]) * scale,
            self.LINK_LENGTH_1 * sin(s[0]) * scale,
        ]

        p2 = [
            p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]) * scale,
            p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1]) * scale,
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - pi / 2, s[0] + s[1] - pi / 2]
        link_lengths = [self.LINK_LENGTH_1 * scale, self.LINK_LENGTH_2 * scale]

        pygame.draw.line(
            surf,
            start_pos=(-2.2 * scale + offset, 1 * scale + offset),
            end_pos=(2.2 * scale + offset, 1 * scale + offset),
            color=(0, 0, 0),
        )

        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            x = x + offset
            y = y + offset

            l, r, t, b = 0, llen, 0.1 * scale, -0.1 * scale
            coords = [(l, b), (l, t), (r, t), (r, b)]
            transformed_coords = []
            for coord in coords:
                coord = pygame.math.Vector2(coord).rotate_rad(th)
                coord = (coord[0] + x, coord[1] + y)
                transformed_coords.append(coord)
            gfxdraw.aapolygon(surf, transformed_coords, (0, 204, 204))
            gfxdraw.filled_polygon(surf, transformed_coords, (0, 204, 204))
            gfxdraw.aacircle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))
            gfxdraw.filled_circle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))


        # Apply counterclockwise rotation to the entire surface
        angle = 90  # Rotation angle in degrees
        rotated_surf = pygame.transform.rotate(surf, angle)

        # Adjust position for the rotated surface
        rotated_rect = rotated_surf.get_rect(center=(250, 250))  # Center

        pygame.draw.line(
            rotated_surf,
            start_pos=(-2.2 * scale + offset, -1 * scale + offset),
            end_pos=(2.2 * scale + offset, -1 * scale + offset),
            color=(0, 0, 0),
        )

        self.screen.blit(rotated_surf, rotated_rect.topleft)

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
def wrap(x, m, M):
    """
    :param x: a scalar
    :param m: minimum possible value in range
    :param M: maximum possible value in range
    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def bound(x, m, M=None):
    """
    :param x: scalar
    Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.
    *y0*
        initial state vector
    *t*
        sample times
    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``
    *args*
        additional arguments passed to the derivative function
    *kwargs*
        additional keyword arguments passed to the derivative function
    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)
    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),))
    else:
        yout = np.zeros((len(t), Ny))

    yout[0] = y0

    for i in np.arange(len(t) - 1):
        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
