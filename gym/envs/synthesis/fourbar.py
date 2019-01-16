"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import os
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.synthesis.utils import normalized_cross_corelation, normalize_angle, signature, parametrize_path
import matplotlib.pyplot as plt
import requests
import tensorflow as tf
import pickle
from gym.envs.synthesis.models import CuriosityAE
from gym.envs.synthesis.ReplayBuffer import Dataset

""" Init => State space : Options
                    1. Relative position of all joints throughout the simulation and Coupler Curve State
                    2. RNN State after inputing time sequence of all lelative position of all joints throughout the simulation and and Coupler Curve State
            Action space : change the locations of moving pivots and coupler point (which also changes link ration implicitly)
    Reset => Restore the mechanism to original position
    Step => Take action and change the internal value of state and return state, and reward
    Rewards => zero Extrinsic reward
"""
URL = "http://localhost:3000/simulation"

def get_simulated_data(joint_state, linkage_type = 'fourbar'):
    if linkage_type == 'fourbar':
        input_text = make_fourbar_input_text(joint_state)
        f = open('ip.txt','w')
        f.write(input_text)
        f.close()
    info = {'data':input_text}
    # sending get request and saving the response as response object
    r = requests.post(url = URL, data = info)
    return np.array(r.json()['joints'])

def make_fourbar_input_text(joint_state):
    txt = 'Posses:\n' + 'Joints:\n'
    txt += '1 0 R Ground\n'
    txt += '%f %f R\n'%(joint_state[2], joint_state[3])
    txt += '%f %f R\n'%(joint_state[0], joint_state[1])
    txt += '0 0 R Ground\n'
    txt += '%f %f R\n'%(joint_state[2], joint_state[3])
    txt += '%f %f R\n'%(joint_state[4], joint_state[5])
    txt += '%f %f R\n'%(joint_state[0], joint_state[1])
    txt += 'Links:\n'
    txt += '1 -> 2 rgba(173, 255, 47, 0.6)\n'
    txt += '2 -> 3 rgba(173, 255, 47, 0.6)\n'
    txt += '4 -> 3 rgba(173, 255, 47, 0.6)\n'
    txt += '5 -> 6 rgba(173, 255, 47, 0.6)\n'
    txt += '6 -> 7 rgba(173, 255, 47, 0.6)\n'
    return txt


class FourBarExplore(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    def __init__(self):
        self.mode = 'path'
        """
            Distance for Each Joint pair => 9 lengths * 100 steps
            Coupler Curve 100 Points
            Action space => 3^3 - 1 (all ratios are same should not be taken) = 26
        """
        high = np.array([10]*9*50)
        low = np.array([-10]*9*50)

        high_act = np.array([10]*6)
        low_act = np.array([-10]*6)
        self.max_steps = 30

        self.action_space = spaces.Box(low_act, high_act)
        self.observation_space = spaces.Box(low, high)

        self.viewer = None
        self.state = None

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111)
        self.line1 = self.ax1.plot(np.arange(1), np.arange(1), '-', label='Coupler curve')[0]
        self.line2 = self.ax1.plot(np.arange(1), np.arange(1), 'o', label='Coupler curve')[0]
        self.line3 = self.ax1.plot(np.arange(1), np.arange(1), 'o', label='Coupler curve')[0]
        self.l1 = self.ax1.plot(np.arange(1), np.arange(1), 'k-')[0]
        self.l2 = self.ax1.plot(np.arange(1), np.arange(1), 'k-')[0]
        self.l3 = self.ax1.plot(np.arange(1), np.arange(1), 'k-')[0]
        self.l4 = self.ax1.plot(np.arange(1), np.arange(1), 'k-')[0]
        self.l5 = self.ax1.plot(np.arange(1), np.arange(1), 'k-')[0]
        self.ax1.axis('equal')
        self.steps_beyond_done = None

        self.input_state = tf.placeholder(dtype=tf.float32, shape=[None, self.observation_space.shape[0]], name='input_state')
        self.batch_size = np.inf
        self.reward_model = CuriosityAE(hidden_units=225)
        if not os.path.exists("./env-data"):
            os.mkdir("./env-data")

        if os.path.isfile("./env-data/dataset.pkl"):
            with open("./env-data/dataset.pkl", 'rb') as f:
                self.dataset = pickle.load(f)
                f.close()
        else:
            self.dataset = Dataset(1000000)

        self.current_curious_loss = tf.reduce_mean(tf.square(self.reward_model.predict(self.input_state) - self.input_state))
        self.reward_model_grads = tf.gradients(self.current_curious_loss, self.reward_model.trainable_vars)
        reward_model_grads_and_vars = zip(self.reward_model_grads, self.reward_model.trainable_vars)
        self.reward_model_train_op = tf.train.AdamOptimizer(learning_rate=0.01).apply_gradients(reward_model_grads_and_vars)
        self.saver = tf.train.Saver(max_to_keep=3)

    def reset(self):
        '''
        Resets link parameters and returns obervation, reward, done flag and info
        Reseting the task
        Taking part (first 70 points) of a random trajectory as target
        This is because 70 points make non trivial path/motion
        self._state = {'fe': [0, 0.5], 'se': [1,1], 'cp':[0.5,1]}
        '''
        self.steps = 0
        #self._state = [0, 0.5, 1, 1, 0.5, 1]
        self._state = [np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
        joint_data = get_simulated_data(self._state, linkage_type='fourbar')
        self.state = self._calc_state(joint_data)

        self.steps_beyond_done = None
        return np.array(self.state)

    def _do_action(self, action):
        self._state = self._state + action*2e-1

    def _calc_state(self, joint_data):
        """ input : List[List[List]]
            output = numpyArray [900]
        """
        try:
            fg = np.reshape(joint_data[0,0,:].copy(), (2,1))
            sg = np.reshape(joint_data[2,0,:].copy(), (2,1))
        except:
            print(joint_data.shape)

        fe = np.zeros((2,50))
        se = np.zeros((2,50))
        cp = np.zeros((2,50))
        theta = np.zeros((50))

        ind = 0.0
        step = (joint_data.shape[1] - 1)/ 49.0
        i = 0
        while round(ind) < joint_data.shape[1]:
            fe[:,i] = joint_data[1,round(ind),:]
            se[:,i] = joint_data[3,round(ind),:]
            cp[:,i] = joint_data[5,round(ind),:]
            theta[i] = np.arctan2(joint_data[5,round(ind),1] - joint_data[1,round(ind),1], joint_data[5,round(ind),0] - joint_data[1,round(ind),0])
            ind += step
            i += 1
            if i == 50:
                break
        state = np.zeros((50, 9))
        self.coupler_curves = cp
        self.theta = theta

        if i == 50:
            state[:, 0] = np.sum((fe - se)**2, axis=0)**0.5
            state[:, 1] = np.sum((se - cp)**2, axis=0)**0.5
            state[:, 2] = np.sum((fe - cp)**2, axis=0)**0.5
            state[:, 3] = np.sum((cp - fg)**2, axis=0)**0.5
            state[:, 4] = np.sum((cp - sg)**2, axis=0)**0.5
            state[:, 5] = np.sum((se - fg)**2, axis=0)**0.5
            state[:, 6] = np.sum((fe - sg)**2, axis=0)**0.5
            state[:, 7] = np.sum((fe - fg)**2, axis=0)**0.5
            state[:, 8] = np.sum((se - sg)**2, axis=0)**0.5
        else:
            logger.warn('Coupler curve points are is less than 50')

        state = np.reshape(state, [1, 50*9])
        if np.std(state) > 1e-3:
            state = (state - np.mean(state))/np.std(state)
        else:
            state = (state - np.mean(state))

        self.fe = fe
        self.se = se

        return state

    def _calc_params(self):
        p1 = np.sum((self._state[0:2])**2)**0.5
        p2 = np.sum((self._state[2:4] - np.array([1, 0]))**2)**0.5
        p3 = np.sum((self._state[2:4] - self._state[0:2])**2)**0.5
        cp_ff = self._state[4:6] - (self._state[0:2] + self._state[2:4])/2
        phi = np.arctan2(self._state[3] - self._state[1], self._state[2] - self._state[0])
        p4 = cp_ff[0]*np.cos(phi) + cp_ff[1]*np.sin(phi)
        p5 = -cp_ff[0]*np.sin(phi) + cp_ff[1]*np.cos(phi)
        self.params = np.array([p1, p2, p3, p4, p5])
        self.crank_angle = np.arctan2(self._state[1], self._state[0])

    def _calc_reward(self, sess):
        curious_loss = sess.run(self.current_curious_loss, feed_dict={self.input_state: self.state})
        return -1.0 /(curious_loss)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        with tf.Session() as sess:
            try:
                self.saver.restore(sess, "./env-data/model.ckpt")
                sess.graph.finalize()
            except:
                sess.run(tf.global_variables_initializer())
                sess.graph.finalize()

            self.steps += 1

            self._do_action(action)

            joint_data = get_simulated_data(self._state, linkage_type='fourbar')
            self.state = self._calc_state(joint_data)

            reward = self._calc_reward(sess)
            done = False
            if self.steps > self.max_steps:
                done = True
            if not done:
                pass
            elif self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            else:
                if self.steps_beyond_done == 0:
                    logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1

            self._calc_params()
            info = {'params':self.params, 'theta': self.theta, 'cp': self.coupler_curves, 'fe': self.fe, 'se': self.se}

            if reward >= -5:
                self.dataset.add(self.state, self.params, self.fe, self.se, self.coupler_curves, self.theta)
                print('Added to the dataset')
                with open("./env-data/dataset.pkl", 'wb') as f:
                    pickle.dump(self.dataset, f)
                    f.close()

            train_batch = self.dataset.getBatch(self.batch_size)
            curious_loss = sess.run(self.current_curious_loss, feed_dict={ self.input_state: train_batch['state'] })
            while curious_loss >= 0.05:
                curious_loss, _ = sess.run((self.current_curious_loss, self.reward_model_train_op), feed_dict={
                    self.input_state: train_batch['state'] })

            self.saver.save(sess, "./env-data/model.ckpt")

        return (self.state, reward, done, info)

    def render(self, mode='human'):
        ''' Should render fourbar and its coupler curve
        '''

        self.line1.set_data(self.coupler_curves[0, :], self.coupler_curves[1, :])
        self.line2.set_data(self.coupler_curves[0, 0], self.coupler_curves[1, 0])
        self.line3.set_data(self.coupler_curves[0, -1], self.coupler_curves[1, -1])
        self.l1.set_data([0,self._state[0]], [0, self._state[1]])
        self.l2.set_data([1,self._state[2]], [0, self._state[3]])
        self.l3.set_data([self._state[0], self._state[2]], [self._state[1], self._state[3]])
        self.l4.set_data([self._state[0], self._state[4]], [self._state[1], self._state[5]])
        self.l5.set_data([self._state[2], self._state[4]], [self._state[3], self._state[5]])


        self.ax1.autoscale_view()
        self.ax1.relim()
        self.ax1.legend(loc='best')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(1)

    def close(self):
        if self.viewer: self.viewer.close()

class FourBarExploreDiscrete(FourBarExplore):
    def __init__(self):
        super(FourBarExploreDiscrete, self).__init__()
        self.action_space = spaces.Discrete(12)
        self.max_steps = 100

    def _do_action(self, action):
        action_panel = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [-1, 0, 0, 0, 0, 0],
                [0, -1, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, -1, 0],
                [0, 0, 0, 0, 0, -1],
                ])
        self._state = self._state + action_panel[action]*5e-2
