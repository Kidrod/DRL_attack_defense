import re
import joblib
import numpy as np
import pandas as pd

######re_alert_rate
df = pd.read_csv(r"data_for_defense_add_more_experiment/restored_samples/restored_more.csv", index_col=None)
X = df["features"].apply(lambda x:re.findall(r"\d+\.?\d*",x)).values.tolist()
x = [list(map(lambda x: int(x), i))  for i in X]

result = []
for i in x:
    detector = joblib.load(r'detector/SVM.pkl')
    result.append(detector.predict([np.asarray(i)]))
print(result.count(1) / len(result))
print(result.count(1))
#
# with open(r"tmp_add_more_experiment/re_alert_rate_more.txt","w") as f:
#     f.write(str(result.count(1) / len(result)))
