"""
Environment for Mechanism Synthesis via Coupler Synthesis
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.synthesis.utils import simulate_fourbar, is_feasible_fourbar, sample_logNormal_link_params
from gym.envs.synthesis.utils import normalized_cross_corelation, normalize_angle, signature
import matplotlib.pyplot as plt


class CouplerMotionSynthesis(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    '''
    Given the coupler motion, the objective is to find moving pivots with interesting trace paths.
    Here, interesting includes circular/elliptical path, straight lines (or fourbar coupler curves)
    '''
    '''
    State would include coupler motion, moving pivot locations, and traced curves.

    1) we can take euclidien curves as state information
    Or 2) apply convolution to the euclidian curves
    3) use signatures of the curves
    4) apply convolution to signatures of the curves
    '''

    def __init__(self):
        self.mode = 'path'
        high = np.concatenate((np.ones((300,))*np.inf, np.ones((200,))*np.inf, np.array([ np.inf, np.inf])))
        low = np.concatenate((-np.ones((300,))*np.inf, -np.ones((200,))*np.inf, np.array([ -np.inf, np.inf])))
        '''state is : [[100, 100, 100]: coupler motion, [2]: r_x, r_y, [100, 100]: moving point'''

        self.action_space = spaces.Discrete(4)
        ''' +x, -x, +y, -y
        '''
        self.observation_space = spaces.Box(low, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111)
        self.line1 = self.ax1.plot(np.arange(1), np.arange(1), 'r-', label='Coupler')[0]
        self.line2 = self.ax1.plot(np.arange(1), np.arange(1), 'b-', label='Moving Point')[0]
        self.line3 = self.ax1.plot(np.arange(1), np.arange(1), 'k-', )[0]
        self.line4 = self.ax1.plot(np.arange(1), np.arange(1), 'g-', )[0]
        self.ax1.axis('equal')

        self.dr = 0.1 # 5 percent change at each action

        self.steps_beyond_done = None

    def reset(self):
        '''
        Resets link parameters and returns obervation, reward, done flag and info
        Reseting the task
        Taking part (first 70 points) of a random trajectory as target
        This is because 70 points make non trivial path/motion
        '''
        self.steps = 0
        '''
        Reinitializing coupler curves
        Evaluating coupler curves
        '''
        params = sample_logNormal_link_params()
        motion = simulate_fourbar(params).curves[0]
        #motion = get_sample_curves_from_dataset()
        op = signature(x=motion[:, 0], y=motion[:, 1], angle=motion[:, 2])
        x_coupler = op['x']
        y_coupler = op['y']
        theta_coupler = op['normalized_angle']

        self.cx1 = 0
        self.cy1 = 0
        r_x = 0
        r_y = 0
        state = self._calculate_state(r_x, r_y, x_coupler, y_coupler, theta_coupler)
        assert state.shape == self.observation_space.shape

        self.steps_beyond_done = None
        self.state = state

        return self.state

    def _calculate_state(self, r_x, r_y, x_coupler, y_coupler, theta_coupler):
        x, y = move_point(r_x, r_y, x_coupler, y_coupler, theta_coupler)

        curves = np.array([x_coupler, y_coupler, theta_coupler, x, y])
        curves = np.reshape(curves, [-1])

        state = np.concatenate((curves,[r_x]))
        state = np.concatenate((state,[r_y]))
        return state


    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.steps += 1

        self._calculate_params(action)

        r_x, r_y = self.state[-2:]
        x = self.state[300:400]
        y = self.state[400:500]
        x_coupler = self.state[0:100]
        y_coupler = self.state[100:200]
        theta_coupler = self.state[200:300]
        state = self._calculate_state(r_x, r_y, x_coupler, y_coupler, theta_coupler)
        self.state = state

        reward, is_sucess = self._evaluate_step()

        '''
        TODO: goal state should be of fixed dimensions, which currently is not.
        for goal based RL algorithms use commented
        '''
        if not is_sucess:
            pass
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1


        return (self.state, reward, is_sucess, {'is_sucess': is_sucess})

    def _calculate_params(self, action):
        ''' Action is changing r_x and r_y
        '''
        r_x, r_y = self.state[-2:]
        if action == 0:
            r_x += self.dr
        if action == 1:
            r_x -= self.dr
        if action == 2:
            r_y += self.dr
        if action == 3:
            r_y -= self.dr

        self.state[-2:] = r_x, r_y

        return r_x, r_y

    def _evaluate_step(self):
        ''' Calculates reward '''
        x = self.state[300:400]
        y = self.state[400:500]
        data = getFittingError(x, y)
        # scaling fitting error to reward signal
        reward = -data['e']*10
        if reward < -40:
            reward = -40 + (reward/40)

        ''' Penalizing movements
        '''
        reward -= 0.02

        try:
            self.r1 = data['r']
            self.cx1 = data['c_x']
            self.cy1 = data['c_y']
        except:
            pass

        if reward >= -0.5:
            success = True
        else:
            success = False

        return reward, success

    def render(self, mode='human'):
        ''' Should render fourbar and its coupler curve
        '''
        x = self.state[300:400]
        y = self.state[400:500]
        x_coupler = self.state[0:100]
        y_coupler = self.state[100:200]
        self.ax1.autoscale_view()
        self.ax1.relim()
        self.line1.set_data(x_coupler, y_coupler)
        self.line2.set_data(x, y)
        self.line3.set_data([x_coupler[0], x[0]], [y_coupler[0], y[0]])
        try:
            self.line4.set_data([self.cx1, x[0]], [self.cy1, y[0]])
        except:
            pass

        self.ax1.legend(loc='best')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)

def move_point(r_x, r_y, x_coupler, y_coupler, theta_coupler):
    x_moving_point = x_coupler + r_x*np.cos(theta_coupler) - r_y*np.sin(theta_coupler)
    y_moving_point = y_coupler + r_y*np.cos(theta_coupler) + r_x*np.sin(theta_coupler)
    return x_moving_point, y_moving_point

def getFittingError(x, y):
    '''
    Algebraic fitting error per point, when N points (x:[N], y:[N])
    are subjected to fit unified equation of line and circle.
    '''
    mat = np.array([x**2 + y**2, -2*x, -2*y, -np.ones(x.shape)])
    mat = np.transpose(mat)

    u, s, v = np.linalg.svd(mat)

    v1 = v[3, :]
    e = np.matmul(mat, v1)
    '''
    Normalizing the error
    '''
    denom = np.sqrt(mat[:,0]**2 + mat[:,1]**2 + mat[:, 2]**2 + mat[:, 3]**2)
    assert denom.shape == mat[:,0].shape

    e = np.sum(np.square(e*denom))
    #e = np.sum(np.square(e))

    e = e/x.shape[0]

    a0 = v1[0]
    a1 = v1[1]
    a2 = v1[2]
    a3 = v1[3]
    r = np.sqrt((a1**2 + a2**2 + a3*a0)/(a0**2))
    c_x = a1/a0
    c_y = a2/a0

    data = dict()

    if e <= 1e-2:
        if r < 100:
            data['dyad_type'] = 'rr'
        if r >= 100 and (np.abs(c_x) > 1000 or np.abs(c_x) > 1000):
            data['dyad_type'] = 'pr'
        else:
            data['dyad_type'] = 'rp'

        data['c_x'] = c_x
        data['c_y'] = c_y
        data['r'] = r

    data['e'] = e
    #print('Center of circle is :({}, {})'.format(a1/a0, a2/a0))
    #print('Radius of circle is :({})'.format(r))
    #print('Fitting error = {}'.format(e))

    return data
