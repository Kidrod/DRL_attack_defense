import re
import pandas as pd
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from gym_ids.envs.features import Features
import joblib
from chuli import get_filelist
import sklearn.svm as svm


############Evaded RF on SVM##############
# df_adversarial = pd.read_csv(r"data_for_defense/evaded_samples_with_origin/evaded.csv", index_col=None)
# X_adversarial = df_adversarial["features"].apply(lambda x:re.findall(r"\d+\.?\d*",x)).values.tolist()
# X_adversarial = [list(map(lambda x: int(x), i))  for i in X_adversarial][500:1500]
#
# y_test = pd.Series([1] * len(X_adversarial))
#
# print("---SVM---")
# clf = joblib.load("detector/SVM.pkl")
# y_pred = clf.predict(X_adversarial)
# print(pd.value_counts(y_pred))
# acs = accuracy_score(y_test, y_pred)
# cm = confusion_matrix(y_test, y_pred)
# print("Accuracy:", acs)
# print("\nConfusion Matrix:\n", cm)
# report = classification_report(y_test, y_pred, output_dict=True, digits=5)
# df_result = pd.DataFrame(report).transpose()
# df_result.to_csv("tmp_add_more_experiment/SVM_result.csv", index=True)



############Evaded SVM on RF##############

df_adversarial = pd.read_csv(r"data_for_defense_add_more_experiment/evaded_samples_with_origin/evaded.csv", index_col=None)
X_adversarial = df_adversarial["features"].apply(lambda x:re.findall(r"\d+\.?\d*",x)).values.tolist()
X_adversarial = [list(map(lambda x: int(x), i))  for i in X_adversarial][500:1500]

y_test = pd.Series([1] * len(X_adversarial))

print("---SVM---")
clf = joblib.load("detector/RF.pkl")
y_pred = clf.predict(X_adversarial)
print(pd.value_counts(y_pred))
acs = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", acs)
print("\nConfusion Matrix:\n", cm)
report = classification_report(y_test, y_pred, output_dict=True, digits=5)
df_result = pd.DataFrame(report).transpose()
df_result.to_csv("tmp_add_more_experiment/RF_result.csv", index=True)
