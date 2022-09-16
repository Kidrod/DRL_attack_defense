import numpy as np
import pandas as pd
from gym_ids.envs.features import Features
import joblib

class Darknet_Check(object):
    def __init__(self):
        self.name="Darknet_Check"
        self.feature = Features()

    def check_darknet(self,sample,model_name = "RF"):
        detector = joblib.load(r'detector/{}.pkl'.format(model_name))
        isxss = detector.predict([self.feature.feature_extraction(sample)])
        return True if isxss[0] == 1 else False

if __name__ == '__main__':
    detector=Darknet_Check()
    k = r"D:\Download\DRL_attack_defense\malicious_train\3477e9667a10efcca41d626b88ec15e3fcda28d8.pdf"
    print(detector.check_darknet(k))