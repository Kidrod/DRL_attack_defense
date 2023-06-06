import numpy as np
import pandas as pd
from gym_ids.envs.features import Features
import joblib

class Darknet_Check(object):
    def __init__(self):
        self.name="Darknet_Check"
        self.feature = Features()

    def check_darknet(self,sample,model_name = "RF"):
        detector = joblib.load(r'/content/DRL_attack_defense/detector/{}.pkl'.format(model_name))
        isxss = detector.predict([self.feature.feature_extraction(sample)])
        return True if isxss[0] == 1 else False

if __name__ == '__main__':
    detector=Darknet_Check()
    k = r"/content/sample_data/Malicious/malicious_train/02ea588af725212ce1f4e8590cd9ce0093fae9ff527446ae80a8f2528c0a4b17"
    print(detector.check_darknet(k))
