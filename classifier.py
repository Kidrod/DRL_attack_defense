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

######Get the path#######
# temp = []
# list_benign = get_filelist(r'D:\Download\DRL_attack_defense\benign_for_detection', [])
# list_malicious = get_filelist(r'D:\Download\DRL_attack_defense\malicious_for_detection', [])[:10000]
# list_all = list_benign + list_malicious
#
# for i in list_all:
#     temp.append(list(Features().feature_extraction(i)))
# df = pd.DataFrame(temp,columns=["obj","endobj","stream","endstream","xref","trailer","startxref","Page","Encrypt","ObjStm",
#                                 "JS","Javascript","AA","OpenAction","AcroForm","JBIG2Decode","RichMedia","Launch","EmbeddedFile","XFA","Colors"])
# df.to_csv("data.csv",index=False)
# y = [0] * len(list_benign) + [1] * len(list_malicious)
# df_label =  pd.DataFrame(y,columns=["label"])
# df_label.to_csv("label.csv",index=False)


df_data = read_csv('data.csv',index_col=None)
df_label = read_csv('label.csv',index_col=None)
X = df_data.iloc[:, 0: 21]
y = df_label.iloc[:, 0]


############Load feature vectors of adversarial samples##############
df_adversarial = pd.read_csv(r"data_for_defense/evaded_samples_with_origin/evaded.csv", index_col=None)
X_adversarial = df_adversarial["features"].apply(lambda x:re.findall(r"\d+\.?\d*",x)).values.tolist()
X_adversarial = [list(map(lambda x: int(x), i))  for i in X_adversarial][:500]

y_adversarial = [1] * 500

df2 = pd.DataFrame(X_adversarial,columns=("obj","endobj","stream","endstream","xref","trailer","startxref","Page","Encrypt","ObjStm","JS","Javascript","AA","OpenAction","AcroForm","JBIG2Decode","RichMedia","Launch","EmbeddedFile","XFA","Colors"))
X = X.append(df2,ignore_index = True)
y = y.append(pd.Series(y_adversarial),ignore_index = True)
########################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


########################################
# print("---Random Forest---")
# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# acs = accuracy_score(y_test, y_pred)
# cm = confusion_matrix(y_test, y_pred)
# print("Accuracy:", acs)
# print("\nConfusion Matrix:\n", cm)
# joblib.dump(clf, 'detector/{}.pkl'.format("RF"))
#
# report = classification_report(y_test, y_pred, output_dict=True, digits=5)
# df_result = pd.DataFrame(report).transpose()
# df_result.to_csv("detector/RF_result.csv", index=True)
#
# # clf = joblib.load("detector/RF.pkl")
#
# # features = feature_extraction(path)
# # print(features)
# # result = clf.predict(features)
# # print(result)


########################################
print("---SVM---")
model = svm.SVC()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acs = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", acs)
print("\nConfusion Matrix:\n", cm)
joblib.dump(model, 'detector/{}.pkl'.format("SVM"))
report = classification_report(y_test, y_pred, output_dict=True, digits=5)
df_result = pd.DataFrame(report).transpose()
df_result.to_csv("detector/SVM_result.csv", index=True)