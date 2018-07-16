"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.synthesis.utils import simulate_fourbar, is_feasible_fourbar, sample_logNormal_link_params
from gym.envs.synthesis.utils import normalized_cross_corelation, normalize_angle, signature
import matplotlib.pyplot as plt

class FourBarPathEnv1(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.mode = 'path'
        high = np.concatenate((np.array([ 5.0, 5.0, 5.0, 5.0, 5.0]), np.ones((100,))*np.inf))
        low = np.concatenate((np.array([ 0.2, 0.2, 0.2, -5.0, -5.0]), -np.ones((100,))*np.inf))

        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111)
        self.line1 = self.ax1.plot(np.arange(1), np.arange(1), 'o', label='Task')[0]
        self.line2 = self.ax1.plot(np.arange(1), np.arange(1), '-', label='Achieved_1')[0]
        self.line3 = self.ax1.plot(np.arange(1), np.arange(1), '-', label='Achieved_2')[0]
        self.line4 = self.ax1.plot(np.arange(1), np.arange(1), '-', label='Achieved_3')[0]
        self.line5 = self.ax1.plot(np.arange(1), np.arange(1), '-', label='Achieved_4')[0]
        #self.l1 = self.ax1.plot(np.arange(1), np.arange(1), '-')[0]
        #self.l2 = self.ax1.plot(np.arange(1), np.arange(1), '-')[0]
        #self.l3 = self.ax1.plot(np.arange(1), np.arange(1), '-')[0]
        #self.l4 = self.ax1.plot(np.arange(1), np.arange(1), '-')[0]
        #self.l5 = self.ax1.plot(np.arange(1), np.arange(1), '-')[0]
        self.ax1.axis('equal')

        self.dl = 0.05 # 5 percent change at each action

        self.steps_beyond_done = None

    def reset(self):
        '''
        Resets link parameters and returns obervation, reward, done flag and info
        Reseting the task
        Taking part (first 70 points) of a random trajectory as target
        This is because 70 points make non trivial path/motion
        '''
        self.steps = 0
        self.goal_params = sample_logNormal_link_params()
        coupler_curves = simulate_fourbar(self.goal_params)
        if len(coupler_curves.signs) == 0:
            print('Params : {} are invalid'.format(self.goal_params))
            print('Coupler points in each curve are : {}, {}, {}, {}'.format(len(coupler_curves.curv1), len(coupler_curves.curv2), len(coupler_curves.curv3), len(coupler_curves.curv4)))
            raise ValueError
        task = coupler_curves.signs[0]
        self.task = task[self.mode + '_sign']
        self.goal = task['fixed_' + self.mode + '_sign'].flatten()
        self.goal_trajectory = coupler_curves.curv1

        '''
        Reinitializing coupler curves
        Evaluating coupler curves
        '''
        self.params = sample_logNormal_link_params()
        self.coupler_curves = simulate_fourbar(self.params)

        self.steps_beyond_done = None

        self.state = np.concatenate((self.params, self.goal), axis=0)

        return np.array(self.state)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.steps += 1

        self._calculate_params(action)

        reward, done = self._evaluate_step()

        is_sucess = done
        if self.steps > 1000:
            done = True
        '''
        TODO: goal state should be of fixed dimensions, which currently is not.
        for goal based RL algorithms use commented
        '''
        if not done:
            pass
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1

        self.state = np.concatenate((self.params, self.goal), axis=0)

        return (self.state, reward, done, {'is_sucess': is_sucess})

    def render(self, mode='human'):
        ''' Should render fourbar and its coupler curve
        '''
        self.ax1.autoscale_view()
        self.ax1.relim()
        self.line1.set_data(self.goal_trajectory[:,0], self.goal_trajectory[:,1])
        if len(self.coupler_curves.curv1) > 5:
            self.line2.set_data(self.coupler_curves.curv1[:,0], self.coupler_curves.curv1[:,1])
        if len(self.coupler_curves.curv2) > 5:
            self.line3.set_data(self.coupler_curves.curv2[:,0], self.coupler_curves.curv2[:,1])
        if len(self.coupler_curves.curv3) > 5:
            self.line4.set_data(self.coupler_curves.curv3[:,0], self.coupler_curves.curv3[:,1])
        if len(self.coupler_curves.curv4) > 5:
            self.line5.set_data(self.coupler_curves.curv4[:,0], self.coupler_curves.curv4[:,1])

        #self.l1.set_data([0,self.state[0]], [0, self.state[1]])
        #self.l2.set_data([1,self.state[2]], [0, self.state[3]])
        #self.l3.set_data([self.state[0], self.state[2]], [self.state[1], self.state[3]])
        #self.l4.set_data([self.state[0], self.state[4]], [self.state[1], self.state[5]])
        #self.l5.set_data([self.state[2], self.state[4]], [self.state[3], self.state[5]])


        self.ax1.legend(loc='best')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)

    def close(self):
        if self.viewer: self.viewer.close()

    def _calculate_params(self, action):
        ''' Action is changing the link ratios
            if actions leads into invalid link combination then it is discarded i.e. returned to previous state.
        '''
        l_diff = [self.dl, self.dl, self.dl, self.dl, self.dl]
        action_vector = np.zeros((5,))
        if action > 4:
            action = action - 5
            action_vector[action] = -1
        else:
            action_vector[action] = 1
        params = self.params + l_diff * action_vector

        if is_feasible_fourbar(params):
            self.params = params

    def _evaluate_step(self):
        ''' Calculates reward for current parameters and task
        '''
        coupler_curves = simulate_fourbar(self.params)
        if len(coupler_curves.signs) == 0:
            print('Params : {} are invalid'.format(self.params))
            print('Coupler points in each curve are : {}, {}, {}, {}'.format(len(coupler_curves.curv1), len(coupler_curves.curv2), len(coupler_curves.curv3), len(coupler_curves.curv4)))
            raise ValueError

        reward = -1
        if self.mode == 'path':
            for sign in coupler_curves.signs:
                result = normalized_cross_corelation(coupler_curves.signs[1][self.mode + '_sign'], self.task)
                if reward < result['score'] - 1:
                    reward = result['score'] - 1


        if reward > -0.02:
            success = True
        else:
            success = False

        if success:
            with open('synth_stats.txt', 'a') as f:
                f.write('Synthesis complete for params : {} is {} with error {}.\n'.format(self.goal_params, self.params, -reward))

        self.coupler_curves = coupler_curves

        return reward, success
