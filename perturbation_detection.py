import pickle
import re
import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance

########split the dataset########
# df_origin = pd.read_csv(r"data_for_defense_add_more_experiment_no/evaded_samples_with_origin/origin.csv",index_col=None)
# df_evaded = pd.read_csv(r"data_for_defense_add_more_experiment_no/evaded_samples_with_origin/evaded.csv",index_col=None)
#
# detection_len = int(len(df_evaded) * 3 / 10)            #split 3:7
#
# df_train_detection = df_evaded[:detection_len]
# df_test_detection = df_evaded[detection_len:]
# df_origin_detection = df_origin[:detection_len]
# df_origin_detection_test = df_origin[detection_len:]
#
# df_train_detection.to_csv(r"data_for_defense_add_more_experiment_no/evaded_samples_with_origin/evaded_for_detection.csv",index = False)
# df_test_detection.to_csv(r"data_for_defense_add_more_experiment_no/evaded_samples_with_origin/evaded_for_attack.csv",index = False)
#
# df_origin_detection.to_csv(r"data_for_defense_add_more_experiment_no/evaded_samples_with_origin/origin_for_detection.csv",index = False)
# df_origin_detection_test.to_csv(r"data_for_defense_add_more_experiment_no/evaded_samples_with_origin/origin_for_attack.csv",index = False)


########Load data########

df_perturbation = pd.read_csv(r"data_for_defense_add_more_experiment/evaded_samples_with_origin/evaded_for_detection.csv",index_col=None)
df_origin = pd.read_csv(r"data_for_defense_add_more_experiment/evaded_samples_with_origin/origin_for_detection.csv",index_col=None)

###add some adversarial samples####

df_perturbation_RF = pd.read_csv(r"data_for_defense/evaded_samples_with_origin/evaded_for_detection.csv",index_col=None)[:100]
df_origin_RF = pd.read_csv(r"data_for_defense/evaded_samples_with_origin/origin_for_detection.csv",index_col=None)[:100]            #load original samples that evaded successfully
df = pd.concat([df_perturbation,df_origin,df_perturbation_RF,df_origin_RF])         #combination


######################

# df = pd.concat([df_perturbation,df_origin])
y = df['type'].values.tolist()
X = df["features"].apply(lambda x:re.findall(r"\d+\.?\d*",x)).values.tolist()
x = [list(map(lambda x: int(x), i))  for i in X]


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=1)

#XGboost
xgb_model = xgb.XGBClassifier(importance_type='weight')
xgb_model.fit(x_train,y_train)
y_pred=xgb_model.predict(x_test)

joblib.dump(xgb_model, 'detector/protect_XGboost_add_more_experiment/XGboost_weight_AL.pkl')

report = classification_report(y_test, y_pred, output_dict=True, digits=5)
df_result = pd.DataFrame(report).transpose()
df_result.to_csv("detector/protect_XGboost_add_more_experiment/XGboost_result_weight_AL.csv", index=True)

xgb_model.get_booster().feature_names = ["obj","endobj","stream","endstream","xref","trailer","sxref","Page","Encrypt","ObjStm","JS","JSscript","AA","OpenA","AcroForm","JBIG2Decode","RichMedia","Launch","EmbeddedFile","XFA","Colors"]
importances = zip(["obj","endobj","stream","endstream","xref","trailer","sxref","Page","Encrypt","ObjStm","JS","JSscript","AA","OpenA","AcroForm","JBIG2Decode","RichMedia","Launch","EmbeddedFile","XFA","Colors"]
, xgb_model.feature_importances_)

plot_importance(xgb_model,title=None,importance_type="weight",show_values = True,max_num_features=4)
plt.savefig("tmp_add_more_experiment/feature_importance_weight_AL.jpeg", dpi=600,bbox_inches='tight')
plt.show()