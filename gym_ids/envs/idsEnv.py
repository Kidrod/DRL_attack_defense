#-*- coding:utf-8 –*-
import os
import numpy as np
from gym import spaces
import gym
from gym_ids.envs.features import Features
from gym_ids.envs.ids import Darknet_Check
from gym_ids.envs.flow_manipulator import flow_Manipulator
from chuli import get_filelist
import shutil

samples_train = get_filelist(r'/content/sample_data/Malicious/malicious_train', [])
samples_train_len = len(samples_train)
samples_test = get_filelist(r'/content/sample_data/Malicious/malicious_test', [])

ACTION_LOOKUP = {i: act for i, act in enumerate(flow_Manipulator.ACTION_TABLE.keys())}

class IDSEnv_v0(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self):
        self.action_space = spaces.Discrete(len(ACTION_LOOKUP))
        self.current_sample = None
        self.max_steps = 10
        self.turns = 0

        self.features_extra = Features()
        self.darknet_checker = Darknet_Check()

        self.flow_manipulator = flow_Manipulator()
        self.reset()

    def step(self, action):
        self.turns += 1
        r = 0
        is_gameover = False

        _action = ACTION_LOOKUP[action]

        self.current_sample = self.flow_manipulator.modify(self.current_sample, _action)


        if os.path.getsize(self.current_sample) == 0:       #Early end of episode if file size is none
            r = 10
            is_gameover = True
            self.observation_space = np.array([0] * 21)
        else:
            if not self.darknet_checker.check_darknet(self.current_sample):

                r = 10
                is_gameover=True

                print  ("Good avoid detection!")
                print(_action)
            elif self.turns >= self.max_steps :
                r = -1
                is_gameover=True
            else:
                r = -1
                is_gameover = False
            self.observation_space=self.features_extra.feature_extraction(self.current_sample)

        return self.observation_space, r,is_gameover,{}

    def reset(self):
        self.turns = 0
        while True:
            self.current_sample = samples_train[np.random.randint(0,samples_train_len)]
            if self.darknet_checker.check_darknet(self.current_sample):
                name_pre = self.current_sample[self.current_sample.rfind('/') + 1:]
                shutil.copy(self.current_sample,r"/content/DRL_attack_defense/malicious_add_more_experiment_no/malicious_train_copy")
                self.current_sample = r"/content/DRL_attack_defense/malicious_add_more_experiment_no/malicious_train_copy/{}".format(name_pre)
                break
        self.observation_space = self.features_extra.feature_extraction(self.current_sample)
        return self.observation_space

    def render(self, mode='human', close=False):
        return
