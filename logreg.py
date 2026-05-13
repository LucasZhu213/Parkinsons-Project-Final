import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import joblib

'''
data = pd.read_csv("MDS-UPDRS_Part_I_09Mar2026.csv")
print(data["PATNO"])
data2 = pd.read_csv("MDS_UPDRS_Part_II__Patient_Questionnaire_09Mar2026.csv")
print(data2['PATNO'])
data3 = pd.read_csv("MDS-UPDRS_Part_IV__Motor_Complications_09Mar2026.csv")
print(data3.columns)
print(data3['PATNO'])
cols_to_use = [col for col in data2.columns if col not in data.columns or col == "PATNO" or col == "INFODT"]
cols_to_use2 = [c for c in [col for col in data3.columns if col not in data.columns or col == "PATNO" or col == "INFODT"] if c not in data2.columns or c == 'PATNO' or c == "INFODT"]
print(cols_to_use2)
full_data = pd.merge(pd.merge(data, data2[cols_to_use], on=["PATNO", 'INFODT'], how="left"), data3[cols_to_use2], on=["PATNO", "INFODT"], how='left')
outcomes = []
full_data = full_data.drop(columns=["PAG_NAME"])
labels = pd.read_csv("LABELS.csv").set_index("Subject")
labels = labels[~labels.index.duplicated(keep='first')]
to_drop = set()
for patno in full_data["PATNO"]:
    if int(patno) in labels.index and labels.loc[int(patno)]['Group'] != 'SWEDD': outcomes.append({"Control":0, "Prodromal":1, "PD":2}[labels.loc[int(patno)]['Group']])
    else: to_drop.add(patno)
keep = []
for i in range(len(full_data)):
    if full_data.iloc[i]['PATNO'] not in to_drop: keep.append(i)
full_data = full_data.iloc[keep]
full_data["OUTCOME"] = outcomes
full_data['EVENT_ID'] = full_data['EVENT_ID'].astype("category")
full_data = full_data.drop(columns=["ORIG_ENTRY","LAST_UPDATE", "REC_ID"])
print(full_data)
for c in full_data['EVENT_ID'].cat.categories:
    vals = []
    for i in range(len(full_data)):
        if full_data.iloc[i]['EVENT_ID'] == c: vals.append(1.0)
        else: vals.append(0.0)
    full_data[f"EVENT_ID={c}"] = vals
full_data = full_data.drop(columns=['EVENT_ID'])
print(full_data)
print(full_data.columns)
print([full_data[c].dtype for c in full_data.columns])
full_data.to_csv("logregdata.csv", index=False)
'''
full_data = pd.read_csv("logregdata.csv")
full_data = full_data.set_index("PATNO")
print(full_data)
mice = IterativeImputer(random_state=42, max_iter=10)
print(full_data)
x, y = full_data.drop(columns=['OUTCOME']), full_data['OUTCOME']
print(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
full_train = x_train; full_train['OUTCOME'] = y_train
full_train['INFODT_dt'] = pd.to_datetime(full_train['INFODT'], format='%m/%Y', errors='coerce')
full_train_sorted = full_train.sort_values(['PATNO', 'INFODT_dt'])
full_train = full_train_sorted.groupby('PATNO').head(5)
print(full_train)
full_train = full_train.drop(columns=["INFODT", "INFODT_dt"])
full_train = pd.DataFrame(
    mice.fit_transform(full_train),
    columns=full_train.columns,
    index=full_train.index,
)
full_test = x_test; full_test['OUTCOME'] = y_test
full_test = full_test.drop(columns=["INFODT"])
full_test = pd.DataFrame(
    mice.transform(full_test),
    columns=full_test.columns,
    index=full_test.index,
)
x_train = full_train.drop(columns=["OUTCOME"]); y_train = full_train['OUTCOME']
full_train.to_csv("logregimptrain.csv", index=True)
full_test.to_csv('logregimptest.csv', index=True)
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(x_train, y_train)
x_test = full_test.drop(columns=['OUTCOME']); y_test = full_test['OUTCOME']
y_pred = lr.predict(x_test)
acc = accuracy_score(y_pred, y_test)
cm = confusion_matrix(y_pred, y_test)
print(acc)
print(cm)
y_pred_train = lr.predict(x_train)
acc_train = accuracy_score(y_pred_train, y_train)
cm_train = confusion_matrix(y_pred_train, y_train)
print(acc_train)
print(cm_train)
joblib.dump(lr, "lr_model.joblib")
