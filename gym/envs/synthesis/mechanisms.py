import os
import math
import gym
from gym import logger
import gym.spaces as spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import requests
import tensorflow as tf
import pickle
from gym.envs.synthesis.models import CuriosityAE
from gym.envs.synthesis.ReplayBuffer import Dataset
import xml.etree.ElementTree as ET
from matplotlib.patches import Circle, Wedge, Polygon

class MechanismExplore(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    def __init__(self, linkage_type, obs_dim, act_dim, z_dim=32):
        self.mode = 'path'
        self.linkage_type = linkage_type
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        high_obs = np.array([10]*obs_dim*100)
        low_obs = np.array([-10]*obs_dim*100)
        high_act = np.array([10]*act_dim)
        low_act = np.array([-10]*act_dim)
        self.max_steps = 30
        self.action_space = spaces.Box(low_act, high_act)
        self.observation_space = spaces.Box(low_obs, high_obs)
        self._render_bool = False
        ''' Internal Model for Measuring Curiosity
        '''
        self.input_state = tf.placeholder(dtype=tf.float32, shape=[None, self.observation_space.shape[0]], name='input_state')
        self.batch_size = np.inf
        self.reward_model = CuriosityAE(name= 'curiousity' + self.linkage_type , hidden_units=z_dim)
        self._load_dataset()

        self.current_curious_loss = tf.reduce_mean(tf.square(self.reward_model.predict(self.input_state) - self.input_state))
        self.reward_model_grads = tf.gradients(self.current_curious_loss, self.reward_model.trainable_vars)
        reward_model_grads_and_vars = zip(self.reward_model_grads, self.reward_model.trainable_vars)
        self.reward_model_train_op = tf.train.AdamOptimizer(learning_rate=0.01).apply_gradients(reward_model_grads_and_vars)
        self.saver = tf.train.Saver(max_to_keep=3)

    def render_init(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('equal')

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        with tf.Session() as sess:
            try:
                self.saver.restore(sess, "./env-data-%s/model.ckpt"%self.linkage_type)
                sess.graph.finalize()
            except:
                sess.run(tf.global_variables_initializer())
                sess.graph.finalize()

            self.steps += 1

            self._do_action(action)

            self.root = self.get_XML_from_state(self._state)
            self.joint_data, self.ip = self.get_simulated_data(self.root)
            self.state = self._calc_state(self.joint_data)

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

            info = self._calc_info()

            if reward >= -5:
                self._add_to_dataset()
                print('Added to the dataset, Dataset Points: %d'%self.dataset.num_experiences)
                self._save_dataset()

            train_batch = self.dataset.getBatch(self.batch_size)
            curious_loss = sess.run(self.current_curious_loss, feed_dict={ self.input_state: train_batch['state'] })
            while curious_loss >= 0.05:
                curious_loss, _ = sess.run((self.current_curious_loss, self.reward_model_train_op), feed_dict={
                    self.input_state: train_batch['state'] })

            self.saver.save(sess, "./env-data-%s/model.ckpt"%self.linkage_type)

        return (self.state, reward, done, info)

    def _calc_reward(self, sess):
        curious_loss = sess.run(self.current_curious_loss, feed_dict={self.input_state: self.state})
        return -1.0 /(curious_loss)


    def reset(self):
        '''
        Resets link parameters and returns obervation, reward, done flag and info
        Reseting the task
        Taking part (first 70 points) of a random trajectory as target
        This is because 70 points make non trivial path/motion
        self._state = {'fe': [0, 0.5], 'se': [1,1], 'cp':[0.5,1]}
        '''
        self.steps = 0
        self._state = self._get_random_state()
        self.root = self.get_XML_from_state(self._state)
        self.joint_data, self.ip = self.get_simulated_data(self.root)
        self.state = self._calc_state(self.joint_data)

        self.steps_beyond_done = None
        return np.array(self.state)

    def _load_dataset(self):
        if not os.path.exists("./env-data-%s"%self.linkage_type):
            os.mkdir("./env-data-%s"%self.linkage_type)

        if os.path.isfile("./env-data-%s/dataset.pkl"%self.linkage_type):
            with open("./env-data-%s/dataset.pkl"%self.linkage_type, 'rb') as f:
                self.dataset = pickle.load(f)
                f.close()
        else:
            self.dataset = Dataset(1000000)

        return self.dataset

    def _save_dataset(self):
       with open("./env-data-%s/dataset.pkl"%self.linkage_type, 'wb') as f:
           pickle.dump(self.dataset, f)

    def close(self):
        if self.viewer: self.viewer.close()

    def render(self, mode='human'):
        if not self._render_bool:
            self.render_init()
            self._render_bool = True
        self.ax.clear()
        self.ax = self.plot_linkage_from_xml(self.root, self.ax)
        self.ax = self.plot_curve(self.coupler_curves, self.ax)

    def plot_linkage_from_xml(self, root, ax):
        for link_id in range(len(root[0])):
            fp = self.getPoint(root, link_id, 0)
            sp = self.getPoint(root, link_id, 1)
            if 'class' in root[0][link_id].attrib and root[0][link_id].attrib['class'] == 'Ternary Link':
                tp = self.getPoint(root, link_id, 2)
                ax = self.draw_triad(fp, sp, tp, ax)
            else:
                ax = self.draw_dyad(fp, sp, ax)
        return ax

    def _add_to_dataset(self):
        experience = {'state':self.state,
                'theta': self.theta,
                'cp': self.coupler_curves,
                'joint_data': self.joint_data,
                'ip': self.ip
                }
        self.dataset.add(experience)

    def _calc_info(self):
        info = {'theta': self.theta, 'cp': self.coupler_curves}
        return info

    @staticmethod
    def plot_curve(cur, ax, label=None):
        ax.plot(cur[:,0], cur[:,1], lw=1.5, color='black', label=label)
        return ax

    @staticmethod
    def get_simulated_data(root):
        URL = "http://localhost:3000/simulation"
        xml_str = ET.tostring(root).decode()
        with open('temp.xml', 'w') as f:
            f.write(xml_str)
            f.close()
        info = {'data':xml_str}
        # sending get request and saving the response as response object
        r = requests.post(url = URL, data = info)
        return np.array(r.json()['joints']), xml_str

    @staticmethod
    def get_abs_cp(fe, se, cp):
        cp_incl = np.arctan2(se[1]-fe[1], se[0]-fe[0])
        cpx = (fe[0]+se[0])/2.0 + cp[0]*np.cos(cp_incl) - cp[1]*np.sin(cp_incl)
        cpy = (fe[1]+se[1])/2.0 + cp[0]*np.sin(cp_incl) + cp[1]*np.cos(cp_incl)
        return cpx, cpy

    @staticmethod
    def draw_triad(pt1, pt2, pt3, ax):
        xy = np.array([pt1, pt2, pt3])
        ax.add_patch(Polygon(xy, closed=True, fill=True, edgecolor='teal', facecolor='lawngreen', alpha=0.3))
        return ax

    @staticmethod
    def draw_dyad(pt1, pt2, ax):
        ax.plot(pt1[0], pt1[1], 'k.', ms=10, alpha=0.5)
        ax.plot(pt2[0], pt2[1], '.', ms=10, alpha=0.5)
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='coral', lw=1.5, alpha=0.3)
        return ax

    @staticmethod
    def getPoint(root, link_id, key_id):
        keys = {0: ('x', 'y'), 1: ('x1', 'y1'), 2: ('x2', 'y2')}
        fp = [float(root[0][link_id].attrib[keys[key_id][0]]), float(root[0][link_id].attrib[keys[key_id][1]])]
        return fp

class FourBarExplore(MechanismExplore):
    def __init__(self):
        super(FourBarExplore, self).__init__('fourbar', obs_dim = 3, act_dim = 6, z_dim=16)

    def get_XML_from_state(self, state):
        cpx, cpy = self.get_abs_cp(state[0:2], state[2:4], state[4:6])
        root = ET.Element("file")
        linkage = ET.SubElement(root, "linkage")
        attrib = {'type':"RR", 'x':"0", 'y':"0", 'x1':"0", 'y1':"0.5", 'isGround':"true", 'rpm':"6", 'rotationDirection':"1", 'plotCurves':"false,false"
                            }

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = "0", "0"
        link.attrib['x1'], link.attrib['y1'] = str(state[0]), str(state[1])

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = "1", "0"
        link.attrib['x1'], link.attrib['y1'] = str(state[2]), str(state[3])

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = str(state[0]), str(state[1])
        link.attrib['x1'], link.attrib['y1'] = str(state[2]), str(state[3])
        link.attrib['x2'], link.attrib['y2'] = str(cpx), str(cpy)
        link.attrib['isGround'] = "false"
        link.attrib['class'] = "Ternary Link"
        link.attrib['plotCurves'] = "false,false,true"
        return root

    def _get_random_state(self):
        fe = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        se = [np.random.uniform(0, 2), np.random.uniform(-1, 1)]
        cp = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        return np.array(fe+se+cp)

    def _do_action(self, action):
        self._state = np.clip(self._state + action*2e-1, -7, 7)

    def _calc_state(self, joint_data):
        """ input : List[List[List]]
            output = numpyArray [900]
        """
        fe = np.zeros((100, 2))
        se = np.zeros((100, 2))
        cp = np.zeros((100, 2))
        theta = np.zeros((100))

        ind = 0.0
        step = (joint_data.shape[1] - 1)/ 99.0
        i = 0
        while round(ind) < joint_data.shape[1]:
            fe[i,:] = joint_data[1,round(ind),:]
            se[i,:] = joint_data[3,round(ind),:]
            cp[i,:] = joint_data[4,round(ind),:]
            theta[i] = np.arctan2(joint_data[4,round(ind),1] - joint_data[1,round(ind),1], joint_data[4,round(ind),0] - joint_data[1,round(ind),0])
            ind += step
            i += 1
            if i == 100:
                break
        state = np.zeros((100, 3))
        self.coupler_curves = cp
        self.theta = theta

        if i == 100:
            state[:, :2] = cp
            state[:, 2] = theta
        else:
            logger.warn('Coupler curve points are is less than 50')

        state = np.reshape(state, [1, 100*3])
        if np.std(state) > 1e-3:
            state = (state - np.mean(state))/np.std(state)
        else:
            state = (state - np.mean(state))

        return state

class FourBarExploreDiscrete(FourBarExplore):
    def __init__(self):
        super(FourBarExploreDiscrete, self).__init__()
        self.action_space = spaces.Discrete(12)
        self.action_panel = np.array([
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

    def _do_action(self, action):
        self._state = self._state + self.action_panel[action]*5e-2

class SixBarExplore_Steph3a(MechanismExplore):
    def __init__(self):
        super(SixBarExplore_Steph3a, self).__init__('sixbar-steph3a', obs_dim = 6, act_dim = 12, z_dim=16)

    def get_XML_from_state(self, state):
        p3x, p3y = self.get_abs_cp(state[0:2], state[2:4], state[4:6])
        p6x, p6y = self.get_abs_cp(state[4:6], state[8:10], state[10:12])
        root = ET.Element("file")
        linkage = ET.SubElement(root, "linkage")
        attrib = {'type':"RR", 'x':"0", 'y':"0", 'x1':"0", 'y1':"0.5", 'isGround':"true", 'rpm':"6", 'rotationDirection':"1", 'plotCurves':"false,false"
                            }

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = "0", "0"
        link.attrib['x1'], link.attrib['y1'] = str(state[0]), str(state[1])

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = "1", "0"
        link.attrib['x1'], link.attrib['y1'] = str(state[2]), str(state[3])

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = str(state[0]), str(state[1])
        link.attrib['x1'], link.attrib['y1'] = str(state[2]), str(state[3])
        link.attrib['x2'], link.attrib['y2'] = str(p3x), str(p3y)
        link.attrib['isGround'] = "false"
        link.attrib['class'] = "Ternary Link"
        link.attrib['plotCurves'] = "false,false,true"

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = str(state[6]), str(state[7])
        link.attrib['x1'], link.attrib['y1'] = str(state[8]), str(state[9])

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = str(state[8]), str(state[9])
        link.attrib['x1'], link.attrib['y1'] = str(p3x), str(p3y)
        link.attrib['x2'], link.attrib['y2'] = str(p6x), str(p6y)
        link.attrib['isGround'] = "false"
        link.attrib['class'] = "Ternary Link"
        link.attrib['plotCurves'] = "false,false,true"

        return root

    def _get_random_state(self):
        fe = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        se = [np.random.uniform(0, 2), np.random.uniform(-1, 1)]
        fbcp = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        tg = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
        te = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
        cp = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        return np.array(fe+se+fbcp+cp+te+tg)

    def _do_action(self, action):
        self._state = np.clip(self._state + action*2e-1, -9, 9)

    def _calc_state(self, joint_data):
        """ input : List[List[List]]
            output = numpyArray [900]
        """
        p1 = np.zeros((100, 2))
        p2 = np.zeros((100, 2))
        p3 = np.zeros((100, 2))
        p4 = np.zeros((100, 2))
        p5 = np.zeros((100, 2))
        p6 = np.zeros((100, 2))
        theta1 = np.zeros((100))
        theta2 = np.zeros((100))

        ind = 0.0
        step = (joint_data.shape[1] - 1)/ 99.0
        i = 0
        while round(ind) < joint_data.shape[1]:
            p1[i,:] = joint_data[1,round(ind),:]
            p2[i,:] = joint_data[3,round(ind),:]
            p3[i,:] = joint_data[4,round(ind),:]
            p4[i,:] = joint_data[5,round(ind),:]
            p5[i,:] = joint_data[6,round(ind),:]
            p6[i,:] = joint_data[7,round(ind),:]
            theta1[i] = np.arctan2(joint_data[4,round(ind),1] - joint_data[1,round(ind),1], joint_data[4,round(ind),0] - joint_data[1,round(ind),0])
            theta2[i] = np.arctan2(joint_data[7,round(ind),1] - joint_data[4,round(ind),1], joint_data[7,round(ind),0] - joint_data[4,round(ind),0])
            ind += step
            i += 1
            if i == 100:
                break
        state = np.zeros((100, 6))
        self.coupler_curves = p6
        self.theta = theta2

        if i == 100:
            state[:, :2] = p3
            state[:, 2] = theta1
            state[:, 3:5] = p6
            state[:, 5] = theta2
        else:
            logger.warn('Coupler curve points are is less than 50')

        state = np.reshape(state, [1, state.shape[0]*state.shape[1]])
        if np.std(state) > 1e-3:
            state = (state - np.mean(state))/np.std(state)
        else:
            state = (state - np.mean(state))

        return state

class SixBarExplore_Steph1(MechanismExplore):
    def __init__(self):
        super(SixBarExplore_Steph1, self).__init__('sixbar-steph1', obs_dim = 6, act_dim = 12, z_dim=16)

    def get_XML_from_state(self, state):
        p6x, p6y = self.get_abs_cp(state[0:2], state[8:10], state[10:12])
        root = ET.Element("file")
        linkage = ET.SubElement(root, "linkage")
        attrib = {'type':"RR", 'x':"0", 'y':"0", 'x1':"0", 'y1':"0.5", 'isGround':"true", 'rpm':"6", 'rotationDirection':"1", 'plotCurves':"false,false"
                            }

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = "0", "0"
        link.attrib['x1'], link.attrib['y1'] = str(state[0]), str(state[1])
        link.attrib['x2'], link.attrib['y2'] = str(state[2]), str(state[3])
        link.attrib['class'] = "Ternary Link"
        link.attrib['plotCurves'] = "false,false,false"

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = "1", "0"
        link.attrib['x1'], link.attrib['y1'] = str(state[4]), str(state[5])
        link.attrib['x2'], link.attrib['y2'] = str(state[6]), str(state[7])
        link.attrib['class'] = "Ternary Link"
        link.attrib['plotCurves'] = "false,false,false"

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = str(state[2]), str(state[3])
        link.attrib['x1'], link.attrib['y1'] = str(state[4]), str(state[5])
        link.attrib['isGround'] = "false"

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = str(state[6]), str(state[7])
        link.attrib['x1'], link.attrib['y1'] = str(state[8]), str(state[9])
        link.attrib['isGround'] = "false"

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = str(state[0]), str(state[1])
        link.attrib['x1'], link.attrib['y1'] = str(state[8]), str(state[9])
        link.attrib['x2'], link.attrib['y2'] = str(p6x), str(p6y)
        link.attrib['isGround'] = "false"
        link.attrib['class'] = "Ternary Link"
        link.attrib['plotCurves'] = "false,false,true"

        return root

    def _get_random_state(self):
        p1 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        p2 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        p3 = [np.random.uniform(0, 2), np.random.uniform(0, 2)]
        p4 = [np.random.uniform(0, 2), np.random.uniform(0, 2)]
        p5 = [np.random.uniform(-3, 3), np.random.uniform(-3, 3)]
        p6 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        return np.array(p1 + p2 + p3 + p4 + p5 + p6)

    def _do_action(self, action):
        self._state = np.clip(self._state + action*2e-1, -9, 9)

    def _calc_state(self, joint_data):
        """ input : List[List[List]]
            output = numpyArray [900]
        """
        p1 = np.zeros((100, 2))
        p2 = np.zeros((100, 2))
        p3 = np.zeros((100, 2))
        p4 = np.zeros((100, 2))
        p5 = np.zeros((100, 2))
        p6 = np.zeros((100, 2))
        theta1 = np.zeros((100))
        theta2 = np.zeros((100))

        ind = 0.0
        step = (joint_data.shape[1] - 1)/ 99.0
        i = 0
        while round(ind) < joint_data.shape[1]:
            p1[i,:] = joint_data[1,round(ind),:]
            p2[i,:] = joint_data[2,round(ind),:]
            p3[i,:] = joint_data[4,round(ind),:]
            p4[i,:] = joint_data[5,round(ind),:]
            p5[i,:] = joint_data[6,round(ind),:]
            p6[i,:] = joint_data[7,round(ind),:]
            theta1[i] = np.arctan2(joint_data[6,round(ind),1] - joint_data[5,round(ind),1], joint_data[6,round(ind),0] - joint_data[5,round(ind),0])
            theta2[i] = np.arctan2(joint_data[7,round(ind),1] - joint_data[1,round(ind),1], joint_data[7,round(ind),0] - joint_data[1,round(ind),0])
            ind += step
            i += 1
            if i == 100:
                break
        state = np.zeros((100, 6))
        self.coupler_curves = p6
        self.theta = theta2
        if i == 100:
            state[:, :2] = p5
            state[:, 2] = theta1
            state[:, 3:5] = p6
            state[:, 5] = theta2
        else:
            logger.warn('Coupler curve points are is less than 50')

        state = np.reshape(state, [1, state.shape[0]*state.shape[1]])
        if np.std(state) > 1e-3:
            state = (state - np.mean(state))/np.std(state)
        else:
            state = (state - np.mean(state))
        return state

class SixBarExplore_Watt1(MechanismExplore):
    def __init__(self):
        super(SixBarExplore_Watt1, self).__init__('sixbar-watt1', obs_dim = 6, act_dim = 12, z_dim=16)

    def get_XML_from_state(self, state):
        p6x, p6y = self.get_abs_cp(state[6:8], state[8:10], state[10:12])
        root = ET.Element("file")
        linkage = ET.SubElement(root, "linkage")
        attrib = {'type':"RR", 'x':"0", 'y':"0", 'x1':"0", 'y1':"0.5", 'isGround':"true", 'rpm':"6", 'rotationDirection':"1", 'plotCurves':"false,false"
                            }

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = "0", "0"
        link.attrib['x1'], link.attrib['y1'] = str(state[0]), str(state[1])

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = "1", "0"
        link.attrib['x1'], link.attrib['y1'] = str(state[2]), str(state[3])
        link.attrib['x2'], link.attrib['y2'] = str(state[4]), str(state[5])
        link.attrib['class'] = "Ternary Link"
        link.attrib['plotCurves'] = "false,false,false"

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = str(state[0]), str(state[1])
        link.attrib['x1'], link.attrib['y1'] = str(state[2]), str(state[3])
        link.attrib['x2'], link.attrib['y2'] = str(state[6]), str(state[7])
        link.attrib['class'] = "Ternary Link"
        link.attrib['plotCurves'] = "false,false,false"
        link.attrib['isGround'] = "false"

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = str(state[4]), str(state[5])
        link.attrib['x1'], link.attrib['y1'] = str(state[8]), str(state[9])
        link.attrib['isGround'] = "false"

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = str(state[6]), str(state[7])
        link.attrib['x1'], link.attrib['y1'] = str(state[8]), str(state[9])
        link.attrib['x2'], link.attrib['y2'] = str(p6x), str(p6y)
        link.attrib['isGround'] = "false"
        link.attrib['class'] = "Ternary Link"
        link.attrib['plotCurves'] = "false,false,true"

        return root

    def _get_random_state(self):
        p1 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        p2 = [np.random.uniform(0, 2), np.random.uniform(0, 2)]
        p3 = [np.random.uniform(0, 2), np.random.uniform(0, 2)]
        p4 = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
        p5 = [np.random.uniform(-1, 3), np.random.uniform(-1, 3)]
        p6 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        return np.array(p1 + p2 + p3 + p4 + p5 + p6)

    def _do_action(self, action):
        self._state = np.clip(self._state + action*2e-1, -9, 9)

    def _calc_state(self, joint_data):
        """ input : List[List[List]]
            output = numpyArray [900]
        """
        p1 = np.zeros((100, 2))
        p2 = np.zeros((100, 2))
        p3 = np.zeros((100, 2))
        p4 = np.zeros((100, 2))
        p5 = np.zeros((100, 2))
        p6 = np.zeros((100, 2))
        theta1 = np.zeros((100))
        theta2 = np.zeros((100))

        ind = 0.0
        step = (joint_data.shape[1] - 1)/ 99.0
        i = 0
        while round(ind) < joint_data.shape[1]:
            p1[i,:] = joint_data[1,round(ind),:]
            p2[i,:] = joint_data[3,round(ind),:]
            p3[i,:] = joint_data[4,round(ind),:]
            p4[i,:] = joint_data[5,round(ind),:]
            p5[i,:] = joint_data[6,round(ind),:]
            p6[i,:] = joint_data[7,round(ind),:]
            theta1[i] = np.arctan2(joint_data[6,round(ind),1] - joint_data[4,round(ind),1], joint_data[6,round(ind),0] - joint_data[4,round(ind),0])
            theta2[i] = np.arctan2(joint_data[7,round(ind),1] - joint_data[5,round(ind),1], joint_data[7,round(ind),0] - joint_data[5,round(ind),0])
            ind += step
            i += 1
            if i == 100:
                break
        state = np.zeros((100, 6))
        self.coupler_curves = p6
        self.theta = theta2
        if i == 100:
            state[:, :2] = p5
            state[:, 2] = theta1
            state[:, 3:5] = p6
            state[:, 5] = theta2
        else:
            logger.warn('Coupler curve points are is less than 50')

        state = np.reshape(state, [1, state.shape[0]*state.shape[1]])
        if np.std(state) > 1e-3:
            state = (state - np.mean(state))/np.std(state)
        else:
            state = (state - np.mean(state))
        return state

class EightBar_Jansons(MechanismExplore):
    def __init__(self):
        super(EightBar_Jansons, self).__init__('eighbar-jansons', obs_dim = 6, act_dim = 12, z_dim=16)

    def get_XML_from_state(self, state):
        p6x, p6y = self.get_abs_cp(state[6:8], state[8:10], state[10:12])
        root = ET.Element("file")
        linkage = ET.SubElement(root, "linkage")
        linkage.attrib['driving'] = "1"
        attrib = {'type':"RR", 'x':"0", 'y':"0", 'x1':"0", 'y1':"0.5", 'isGround':"true", 'rpm':"6", 'rotationDirection':"1", 'plotCurves':"false,false"
                            }

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = "0", "0"
        link.attrib['x1'], link.attrib['y1'] = str(state[0]), str(state[1])
        link.attrib['x2'], link.attrib['y2'] = str(state[2]), str(state[3])
        link.attrib['class'] = "Ternary Link"
        link.attrib['plotCurves'] = "false,false,false"

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = "1", "0"
        link.attrib['x1'], link.attrib['y1'] = str(state[4]), str(state[5])

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = str(state[4]), str(state[5])
        link.attrib['x1'], link.attrib['y1'] = str(state[6]), str(state[7])
        link.attrib['isGround'] = "false"

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = str(state[4]), str(state[5])
        link.attrib['x1'], link.attrib['y1'] = str(state[0]), str(state[1])
        link.attrib['isGround'] = "false"

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = "0", "0"
        link.attrib['x1'], link.attrib['y1'] = str(state[6]), str(state[7])
        link.attrib['isGround'] = "false"

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = str(state[2]), str(state[3])
        link.attrib['x1'], link.attrib['y1'] = str(state[8]), str(state[9])
        link.attrib['isGround'] = "false"

        link = ET.SubElement(linkage, "link")
        link.attrib = attrib.copy()
        link.attrib['x'], link.attrib['y'] = str(state[6]), str(state[7])
        link.attrib['x1'], link.attrib['y1'] = str(state[8]), str(state[9])
        link.attrib['x2'], link.attrib['y2'] = str(p6x), str(p6y)
        link.attrib['isGround'] = "false"
        link.attrib['class'] = "Ternary Link"
        link.attrib['plotCurves'] = "false,false,true"

        return root

    def _get_random_state(self):
        p1 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        p2 = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
        p3 = [np.random.uniform(0, 2), np.random.uniform(0, 2)]
        p4 = [np.random.uniform(-3, 3), np.random.uniform(-3, 3)]
        p5 = [np.random.uniform(-4, 2), np.random.uniform(-4, 2)]
        p6 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        return np.array(p1 + p2 + p3 + p4 + p5 + p6)

    def _do_action(self, action):
        self._state = np.clip(self._state + action*2e-1, -9, 9)

    def _calc_state(self, joint_data):
        """ input : List[List[List]]
            output = numpyArray [900]
        """
        p1 = np.zeros((100, 2))
        p2 = np.zeros((100, 2))
        p3 = np.zeros((100, 2))
        p4 = np.zeros((100, 2))
        p5 = np.zeros((100, 2))
        p6 = np.zeros((100, 2))
        theta1 = np.zeros((100))
        theta2 = np.zeros((100))

        ind = 0.0
        step = (joint_data.shape[1] - 1)/ 99.0
        i = 0
        while round(ind) < joint_data.shape[1]:
            p1[i,:] = joint_data[1,round(ind),:]
            p2[i,:] = joint_data[2,round(ind),:]
            p3[i,:] = joint_data[4,round(ind),:]
            p4[i,:] = joint_data[5,round(ind),:]
            p5[i,:] = joint_data[6,round(ind),:]
            p6[i,:] = joint_data[7,round(ind),:]
            theta1[i] = np.arctan2(joint_data[6,round(ind),1] - joint_data[2,round(ind),1], joint_data[6,round(ind),0] - joint_data[2,round(ind),0])
            theta2[i] = np.arctan2(joint_data[7,round(ind),1] - joint_data[5,round(ind),1], joint_data[7,round(ind),0] - joint_data[5,round(ind),0])
            ind += step
            i += 1
            if i == 100:
                break
        state = np.zeros((100, 6))
        self.coupler_curves = p6
        self.theta = theta2
        if i == 100:
            state[:, :2] = p5
            state[:, 2] = theta1
            state[:, 3:5] = p6
            state[:, 5] = theta2
        else:
            logger.warn('Coupler curve points are is less than 50')

        state = np.reshape(state, [1, state.shape[0]*state.shape[1]])
        if np.std(state) > 1e-3:
            state = (state - np.mean(state))/np.std(state)
        else:
            state = (state - np.mean(state))
        return state
