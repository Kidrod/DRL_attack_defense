#-*- coding:utf-8 –*-
import re
import joblib
import random
import numpy as np
import pandas as pd
from gym import spaces
import gym
from gym_restore.envs.features import Restore_Features
from gym_restore.envs.ids import Darknet_Check
from gym_restore.envs.flow_manipulator import flow_Manipulator
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"data_for_defense_add_more_experiment/evaded_samples_with_origin/evaded_for_attack.csv", index_col=None)
X = df["features"].apply(lambda x:re.findall(r"\d+\.?\d*",x)).values.tolist()
X = [list(map(lambda x: int(x), i))  for i in X]

# split train:test 6:4
samples_train = X[:int(len(X) * 6 / 10)]
samples_test = X[int(len(X) * 6 / 10):]

df_origin = pd.read_csv(r"data_for_defense_add_more_experiment/evaded_samples_with_origin/origin_for_attack.csv", index_col=None)
X_origin = df_origin["features"].apply(lambda x:re.findall(r"\d+\.?\d*",x)).values.tolist()
origin_for_attack = [list(map(lambda x: int(x), i))  for i in X_origin]
origin_for_attack_data = origin_for_attack[int(len(X) * 6 / 10):]

ACTION_LOOKUP = {i: act for i, act in enumerate(flow_Manipulator.ACTION_TABLE.keys())}

class IDSEnv_v0(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self):
        self.action_space = spaces.Discrete(len(ACTION_LOOKUP))
        self.current_sample = None
        self.max_steps = 50
        self.turns = 0

        self.features_extra = Restore_Features()
        self.darknet_checker = Darknet_Check()

        self.flow_manipulator = flow_Manipulator()
        self.reset()

    def step(self, action):
        self.turns += 1
        r = 0
        is_gameover = False

        _action = ACTION_LOOKUP[action]

        self.current_sample = self.flow_manipulator.modify(self.current_sample, _action)


        if not self.darknet_checker.check_darknet(self.current_sample):

            r = 50
            is_gameover=True

            print  ("Success!")
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
            self.current_sample = [random.choice(samples_train)]
            if self.darknet_checker.check_darknet(self.current_sample):
                self.current_sample = self.current_sample
                break
        self.observation_space = self.features_extra.feature_extraction(self.current_sample)
        return self.observation_space

    def render(self, mode='human', close=False):
        return
