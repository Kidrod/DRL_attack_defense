import re
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine

df_orgin = pd.read_csv(r"data_for_defense_add_more_experiment/restored_samples/origin_malicious_more.csv", index_col=None)
X = df_orgin["features"].apply(lambda x:re.findall(r"\d+\.?\d*",x)).values.tolist()
X_origin = [list(map(lambda x: int(x), i))[4:]  for i in X]             #remove potential influence factors

df_restored = pd.read_csv(r"data_for_defense_add_more_experiment/restored_samples/restored_more.csv", index_col=None)
X = df_restored["features"].apply(lambda x:re.findall(r"\d+\.?\d*",x)).values.tolist()
X_restored = [list(map(lambda x: int(x), i))[4:]  for i in X]

cos,pear_r= 0,0
for i,j in zip(X_origin,X_restored):
    cos += cosine(i,j)
    pear_r += stats.pearsonr(i, j)[0]

print("cosine:" + str(cos / len(X_restored)) + "\n")
print("pear_r:" + str(pear_r / len(X_restored)) + "\n")


with open(r"tmp_add_more_experiment/similarity_more.txt","w") as f:
    f.write("cosine:" + str(cos / len(X_restored)) + "\n")
    f.write("pear_r:" + str(pear_r / len(X_restored)) + "\n")
