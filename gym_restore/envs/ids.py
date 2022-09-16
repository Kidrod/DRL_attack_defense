import re
import pandas as pd
import joblib

class Darknet_Check(object):
    def __init__(self):
        self.name="Darknet_Check"

    def check_darknet(self,sample):
        detector = joblib.load(r'detector/protect_XGboost_add_more_experiment/XGboost_weight_AL.pkl')
        isxss = detector.predict(sample)
        return True if isxss[0] == 2 else False

if __name__ == '__main__':
    detector=Darknet_Check()
    df = pd.read_csv(r"../../data_for_defense/evaded_samples_with_origin/evaded_for_attack.csv", index_col=None)
    X = df["features"].apply(lambda x: re.findall(r"\d+\.?\d*", x)).values.tolist()
    X = [list(map(lambda x: int(x), i)) for i in X]

    print(detector.check_darknet([[-1] * 21]))