import pandas as pd
# import numpy as np
import random as rnd
import time
# visualization
# import seaborn as sns
# import matplotlib.pyplot as plt
# %matplotlib inline
import pickle
from sklearn.preprocessing import MinMaxScaler

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, recall_score, confusion_matrix

# from sklearn import metrics
data_train = pd.read_csv("/Users/sandeepreddygopu/Desktop/Mini Project/mini project.py/TrainData.csv")
data_test = pd.read_csv("/Users/sandeepreddygopu/Desktop/Mini Project/mini project.py/TestData.csv")


print(data_train.shape)
print(data_test.shape)

#data['RIO_ACCEPT_REJECT'] = data['RIO_ACCEPT_REJECT'].astype('category')

'''
colnamesX=['ADDR3MINLENSTRENGTH',  'HNO3NULLSTRENGTH','HNO3STRENGTH',
           'ADDR3MAXLENSTRENGTH',
           'address','city','dob','driving_licn_no','employername','fwacctno',
 'fwcustid','lastname','loanno','mobile','name','pan',  'passport_no','phone','pincode']
    #,'COL1040STRENGTH','COL1050STRENGTH','loanno']
colnames=list(set(data.columns).difference(set(colnamesX)))


colnames = ['address', 'city', 'dob', 'employername', 'lastname',
            'mobile', 'name', 'phone', 'pincode', 'RIO_ACCEPT_REJECT']
data = data[[c for c
             in colnames
             if len(data[c].unique()) > 1]]
'''
#data = data[[c for c
           #  in list(data)
            # if len(data[c].unique()) > 1]]

#ncol = len(data.columns) - 1
#print("No. of features:", ncol)
#print(data.columns)
# test train splitting
# from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split

#train, test = train_test_split(data, stratify=data.RIO_ACCEPT_REJECT.tolist(), test_size=0.3, random_state=13)

X_train = data_train.loc[:, data_train.columns != 'RIO_ACCEPT_REJECT']
Y_train = data_train.RIO_ACCEPT_REJECT
X_test = data_test.loc[:, data_train.columns != 'RIO_ACCEPT_REJECT']
Y_test = data_test.RIO_ACCEPT_REJECT

from sklearn import preprocessing
X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)

'''
scaler = MinMaxScaler().fit(X_train)
with open("/Users/sanndeepreddygopu/Desktop/Mini Project/mini project.py/scaler.pkl", 'wb') as file:  
    pickle.dump(scaler, file)

#with open("/home/mythri/PycharmProjects/FinalClassification/SavedModels/scaler.pkl", 'rb') as file:  
#    scaler2=pickle.load(file)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
'''
# Technique="SMOTE"
# Technique="NORMAL"
# Technique="CLASS_WEIGHT(1:10)"
# Technique="CLASS_WEIGHT(1:15)"
# Technique="CLASS_WEIGHT(1:20)"
#Technique = "SMOTE_CLASS_WEIGHT(1:5)"

# Technique="SMOTE_CLASS_WEIGHT(1:10)"
#Technique="SMOTE_CLASS_WEIGHT(1:15)"
Technique="SMOTE_CLASS_WEIGHT(1:10)"

if (Technique == "NORMAL" or Technique == "SMOTE"):
    classweight = {1: 1}

if ("CLASS_WEIGHT(1:5)" in Technique):
    classweight = {1: 5}

if ("CLASS_WEIGHT(1:10)" in Technique):
    classweight = {1: 10}

if ("CLASS_WEIGHT(1:15)" in Technique):
    classweight = {1: 15}

if ("CLASS_WEIGHT(1:20)" in Technique):
    classweight = {1: 20}

#classweight="balanced"
print(Technique)
print(classweight)
if ("SMOTE" in Technique):
    # SMOTE
    # cols=train.ix[:, train.columns != 'RIO_ACCEPT_REJECT'].columns
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=12, ratio="auto")
    # x_train=train.ix[:, train.columns != 'RIO_ACCEPT_REJECT'].astype(float)
    # y_train=train.RIO_ACCEPT_REJECT
    X_train, Y_train = sm.fit_sample(X_train, Y_train)
'''
    X_train = pd.DataFrame(X_trainSM, columns=X_train.columns)
    Y_train = pd.DataFrame(Y_trainSM, columns=["RIO_ACCEPT_REJECT"])
    train = pd.concat([X_train, Y_train], axis=1)
    Y_train = train.RIO_ACCEPT_REJECT
    print(train.RIO_ACCEPT_REJECT.value_counts())
'''
'''
# code for feature selection

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
model=SelectKBest(chi2, k=32).fit(X_train, Y_train)
#get selected feature names
col_names = list(X_train.columns[model.get_support(indices=True)])
print(col_names)
#get removed features
print("removed cols:",list(set(X_train.columns).difference(set(col_names))))
X_train = model.transform(X_train)
X_test = model.transform(X_test)
#or


#feature reduction
from sklearn.svm import LinearSVC
model1 = LinearSVC(C=0.1, penalty="l1", dual=False,random_state=0).fit(X_train, Y_train)

# code for feature selection
from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(model1, prefit=True)
#get selected feature names
col_names = list(X_train.columns[model.get_support(indices=True)])
print(col_names)
#get removed features
print("******removed cols********\n",list(set(X_train.columns).difference(set(col_names))))
X_train = model.transform(X_train)
X_test = model.transform(X_test)

print("No. of new features:",X_train.shape)
#print(X_train.dtype.names)
'''
# print("Lost features:",set(data.columns).difference(set(X_train.dtype.names)))
result = pd.DataFrame(columns=["ModelName", "Technique", "Class_Weight",
                               "NM_Precision", "NM_Recall",
                               "M_Precision", "M_Recall",
                               "NM_Samples", "M_Samples",
                               "TotalSamples", "VerificationFactor",
                               "TrueNM", "FalseM", "FalseNM", "TrueM"])

    ##############  LOGISTIC REGRESSION AND RESULTS ############
# --------------------------------------------------------------------
print("********************Logistic Results***************")
ModelName = "LogisticRegression"
time0 = time.time()
model1 = LogisticRegression(C=100, dual=False, penalty='l2'
                            , class_weight=classweight).fit(X_train, Y_train)
with open("/Users/sandeepreddygopu/Desktop/Mini Project/mini project.py/Logistic_Regression.pkl", 'wb') as file:  
    pickle.dump(model1, file)

# ,class_weight={1:10}
# code for feature selection
# from sklearn.feature_selection import SelectFromModel
# model = SelectFromModel(model1, prefit=True)
# X_train = model.transform(X_train)
# X_test = model.transform(X_test)
# model1 = LogisticRegression(random_state=0,C=100,penalty='l2').fit(X_train, Y_train)


test_predictions = model1.predict(X_test)
train_predictions = model1.predict(X_train)

print("train results:")
cm_train = confusion_matrix(Y_train, train_predictions)
print(cm_train)

cr_train = classification_report(Y_train, train_predictions)
print(cr_train)
print(accuracy_score(Y_train, train_predictions))

print("test results:")
cm_test = confusion_matrix(Y_test, test_predictions)
print(cm_test)

cr_test = classification_report(Y_test, test_predictions)
print(cr_test)
print(accuracy_score(Y_test, test_predictions))
cm = pd.crosstab(test['RIO_ACCEPT_REJECT'], test_predictions, rownames=["Actual"], colnames=["predicted"])
r = cr_test.split()
VF = (cm.loc[0, 1] + cm.loc[1, 1]) / int(r[20])
print("VF:  ", VF)
result = result.append({"ModelName": ModelName, "Technique": Technique, "Class_Weight": classweight,
                        "NM_Precision": r[5], "NM_Recall": r[6],
                        "M_Precision": r[10], "M_Recall": r[11],
                        "NM_Samples": r[8], "M_Samples": r[13],
                        "TotalSamples": r[20], "VerificationFactor": VF,
                        "TrueNM": cm.loc[0, 0], "FalseM": cm.loc[0, 1],
                        "FalseNM": cm.loc[1, 0], "TrueM": cm.loc[1, 1]},
                       ignore_index=True)
print("Time:", time.time() - time0)

# cm_norm = cm / cm.sum(axis=1)
# cm_norm.to_csv('cm_norm_logistic.csv')
# cm.to_csv('cm_logistic.csv')
# cnf_matrix

# heat map
# from matplotlib import pyplot as plt
# import seaborn as sn
# %matplotlib inline
# plt.figure(figsize=(20, 15))
# sn.set(font_scale=1.7)
# sn.heatmap(cm, annot=True, annot_kws={"size":15}, fmt=".0f")
# plt.show()

#############################
#       RANDOM FORESTS
# ----------------------------------------
ModelName = "RandomForest"

print("****************Random Forest Results***********")
time0 = time.time()

from sklearn.ensemble import RandomForestClassifier

rnd.seed(10)
model1 = RandomForestClassifier(n_estimators=100, random_state=10,
                                max_depth=100, max_features="sqrt", n_jobs=-1, class_weight=classweight).fit(X_train,
                                                                                                             Y_train)
test_predictions = model1.predict(X_test)
train_predictions = model1.predict(X_train)

print("train results:")
cm_train = confusion_matrix(Y_train, train_predictions)
print(cm_train)

cr_train = classification_report(Y_train, train_predictions)
print(cr_train)
print(accuracy_score(Y_train, train_predictions))

print("test results:")
cm_test = confusion_matrix(Y_test, test_predictions)
print(cm_test)

cr_test = classification_report(Y_test, test_predictions)
print(cr_test)
print(accuracy_score(Y_test, test_predictions))
cm = pd.crosstab(test['RIO_ACCEPT_REJECT'], test_predictions, rownames=["Actual"], colnames=["predicted"])
r = cr_test.split()
VF = (cm.loc[0, 1] + cm.loc[1, 1]) / int(r[20])
print("VF:  ", VF)
result = result.append({"ModelName": ModelName, "Technique": Technique, "Class_Weight": classweight,
                        "NM_Precision": r[5], "NM_Recall": r[6],
                        "M_Precision": r[10], "M_Recall": r[11],
                        "NM_Samples": r[8], "M_Samples": r[13],
                        "TotalSamples": r[20], "VerificationFactor": VF,
                        "TrueNM": cm.loc[0, 0], "FalseM": cm.loc[0, 1],
                        "FalseNM": cm.loc[1, 0], "TrueM": cm.loc[1, 1]},
                       ignore_index=True)
print("Time:", time.time() - time0)
# cm_norm = cm / cm.sum(axis=1)
# cm_norm.to_csv('cm_norm_logistic.csv')
# cm.to_csv('cm_logistic.csv')
# cnf_matrix

# heat map
# from matplotlib import pyplot as plt
# import seaborn as sn
# %matplotlib inline
# plt.figure(figsize=(20, 15))
# sn.set(font_scale=1.7)
# sn.heatmap(cm, annot=True, annot_kws={"size":15}, fmt=".0f")
# plt.show()



####################################################
#       PASSIVE AGGRESSIVE CLASSIFIER
# --------------------------------------------

ModelName = "PassiveAgressiveClassifier"

print("****************PAC RESULTS****************")
time0 = time.time()

from sklearn.linear_model import PassiveAggressiveClassifier

model1 = PassiveAggressiveClassifier(C=1.0, average=False,
                                     fit_intercept=True, loss='hinge', max_iter=50, n_iter=None,
                                     n_jobs=1, random_state=0, shuffle=True, verbose=0,
                                     warm_start=False, class_weight=classweight).fit(X_train, Y_train)
test_predictions = model1.predict(X_test)
train_predictions = model1.predict(X_train)

print("train results:")
cm_train = confusion_matrix(Y_train, train_predictions)
print(cm_train)

cr_train = classification_report(Y_train, train_predictions)
print(cr_train)
print(accuracy_score(Y_train, train_predictions))

print("test results:")
cm_test = confusion_matrix(Y_test, test_predictions)
print(cm_test)

cr_test = classification_report(Y_test, test_predictions)
print(cr_test)
print(accuracy_score(Y_test, test_predictions))
cm = pd.crosstab(test['RIO_ACCEPT_REJECT'], test_predictions, rownames=["Actual"], colnames=["predicted"])
r = cr_test.split()
VF = (cm.loc[0, 1] + cm.loc[1, 1]) / int(r[20])
print("VF:  ", VF)
result = result.append({"ModelName": ModelName, "Technique": Technique, "Class_Weight": classweight,
                        "NM_Precision": r[5], "NM_Recall": r[6],
                        "M_Precision": r[10], "M_Recall": r[11],
                        "NM_Samples": r[8], "M_Samples": r[13],
                        "TotalSamples": r[20], "VerificationFactor": VF,
                        "TrueNM": cm.loc[0, 0], "FalseM": cm.loc[0, 1],
                        "FalseNM": cm.loc[1, 0], "TrueM": cm.loc[1, 1]},
                       ignore_index=True)
print("Time:", time.time() - time0)

###############################################
#       PERCEPTRON
# great recall scores but very poor precision scores
# ---------------------------------------

ModelName = "Perceptron"

print("***************Perceptron Results***************")
time0 = time.time()

from sklearn.linear_model import Perceptron
from sklearn.preprocessing import PolynomialFeatures

# X_train = PolynomialFeatures(interaction_only=True).fit_transform(X_train).astype(float)
# X_test = PolynomialFeatures(interaction_only=True).fit_transform(X_test).astype(float)

model1 = Perceptron(fit_intercept=True,
                    shuffle=True, max_iter=100, class_weight=classweight).fit(X_train, Y_train)

test_predictions = model1.predict(X_test)
train_predictions = model1.predict(X_train)

print("train results:")
cm_train = confusion_matrix(Y_train, train_predictions)
print(cm_train)

cr_train = classification_report(Y_train, train_predictions)
print(cr_train)
print(accuracy_score(Y_train, train_predictions))

print("test results:")
cm_test = confusion_matrix(Y_test, test_predictions)
print(cm_test)

cr_test = classification_report(Y_test, test_predictions)
print(cr_test)
print(accuracy_score(Y_test, test_predictions))
cm = pd.crosstab(test['RIO_ACCEPT_REJECT'], test_predictions, rownames=["Actual"], colnames=["predicted"])
r = cr_test.split()
VF = (cm.loc[0, 1] + cm.loc[1, 1]) / int(r[20])
print("VF:  ", VF)
result = result.append({"ModelName": ModelName, "Technique": Technique, "Class_Weight": classweight,
                        "NM_Precision": r[5], "NM_Recall": r[6],
                        "M_Precision": r[10], "M_Recall": r[11],
                        "NM_Samples": r[8], "M_Samples": r[13],
                        "TotalSamples": r[20], "VerificationFactor": VF,
                        "TrueNM": cm.loc[0, 0], "FalseM": cm.loc[0, 1],
                        "FalseNM": cm.loc[1, 0], "TrueM": cm.loc[1, 1]},
                       ignore_index=True)
print("Time:", time.time() - time0)
    
    
    
    
'''
if (classweight == {1: 1}):
    ######################################################
    #    GAUSSIAN NAIVE BAYES
    # -----------------------------------------------
    ModelName = "GaussianNaiveBayes"

    time0 = time.time()

    print("********************GAUSSIAN NAIVE BAYES linear Results***************")
    from sklearn.naive_bayes import GaussianNB

    model1 = GaussianNB().fit(X_train, Y_train)
    test_predictions = model1.predict(X_test)
    train_predictions = model1.predict(X_train)

    print("train results:")
    cm_train = confusion_matrix(Y_train, train_predictions)
    print(cm_train)

    cr_train = classification_report(Y_train, train_predictions)
    print(cr_train)
    print(accuracy_score(Y_train, train_predictions))

    print("test results:")
    cm_test = confusion_matrix(Y_test, test_predictions)
    print(cm_test)

    cr_test = classification_report(Y_test, test_predictions)
    print(cr_test)
    print(accuracy_score(Y_test, test_predictions))
    cm = pd.crosstab(test['RIO_ACCEPT_REJECT'], test_predictions, rownames=["Actual"], colnames=["predicted"])
    r = cr_test.split()
    VF = (cm.loc[0, 1] + cm.loc[1, 1]) / int(r[20])
    print("VF:  ", VF)
    result = result.append({"ModelName": ModelName, "Technique": Technique, "Class_Weight": classweight,
                            "NM_Precision": r[5], "NM_Recall": r[6],
                            "M_Precision": r[10], "M_Recall": r[11],
                            "NM_Samples": r[8], "M_Samples": r[13],
                            "TotalSamples": r[20], "VerificationFactor": VF,
                            "TrueNM": cm.loc[0, 0], "FalseM": cm.loc[0, 1],
                            "FalseNM": cm.loc[1, 0], "TrueM": cm.loc[1, 1]},
                           ignore_index=True)
    print("Time:", time.time() - time0)

    ######################################################
    #       MULTINOMIAL NAIVE BAYES
    # -----------------------------------------------
    ModelName = "MultinomialNaiveBayes"

    time0 = time.time()

    print("********************MULTINOMIAL NAIVE BAYES linear Results***************")
    from sklearn.naive_bayes import MultinomialNB

    model1 = MultinomialNB().fit(X_train, Y_train)
    test_predictions = model1.predict(X_test)
    train_predictions = model1.predict(X_train)

    print("train results:")
    cm_train = confusion_matrix(Y_train, train_predictions)
    print(cm_train)

    cr_train = classification_report(Y_train, train_predictions)
    print(cr_train)
    print(accuracy_score(Y_train, train_predictions))

    print("test results:")
    cm_test = confusion_matrix(Y_test, test_predictions)
    print(cm_test)

    cr_test = classification_report(Y_test, test_predictions)
    print(cr_test)
    print(accuracy_score(Y_test, test_predictions))
    cm = pd.crosstab(test['RIO_ACCEPT_REJECT'], test_predictions, rownames=["Actual"], colnames=["predicted"])
    r = cr_test.split()
    VF = (cm.loc[0, 1] + cm.loc[1, 1]) / int(r[20])
    print("VF:  ", VF)
    result = result.append({"ModelName": ModelName, "Technique": Technique, "Class_Weight": classweight,
                            "NM_Precision": r[5], "NM_Recall": r[6],
                            "M_Precision": r[10], "M_Recall": r[11],
                            "NM_Samples": r[8], "M_Samples": r[13],
                            "TotalSamples": r[20], "VerificationFactor": VF,
                            "TrueNM": cm.loc[0, 0], "FalseM": cm.loc[0, 1],
                            "FalseNM": cm.loc[1, 0], "TrueM": cm.loc[1, 1]},
                           ignore_index=True)
    print("Time:", time.time() - time0)

    ######################################################
    #         AdaBoost Classification

    ModelName = "AdaBoost"

    print("**************AdaBoost Results***************")
    time0 = time.time()

    from sklearn.ensemble import AdaBoostClassifier

    model1 = AdaBoostClassifier(n_estimators=50, random_state=12).fit(X_train, Y_train)

    test_predictions = model1.predict(X_test)
    train_predictions = model1.predict(X_train)

    print("train results:")
    cm_train = confusion_matrix(Y_train, train_predictions)
    print(cm_train)

    cr_train = classification_report(Y_train, train_predictions)
    print(cr_train)
    print(accuracy_score(Y_train, train_predictions))

    print("test results:")
    cm_test = confusion_matrix(Y_test, test_predictions)
    print(cm_test)

    cr_test = classification_report(Y_test, test_predictions)
    print(cr_test)
    print(accuracy_score(Y_test, test_predictions))
    cm = pd.crosstab(test['RIO_ACCEPT_REJECT'], test_predictions, rownames=["Actual"], colnames=["predicted"])
    r = cr_test.split()
    VF = (cm.loc[0, 1] + cm.loc[1, 1]) / int(r[20])
    print("VF:  ", VF)
    result = result.append({"ModelName": ModelName, "Technique": Technique, "Class_Weight": classweight,
                            "NM_Precision": r[5], "NM_Recall": r[6],
                            "M_Precision": r[10], "M_Recall": r[11],
                            "NM_Samples": r[8], "M_Samples": r[13],
                            "TotalSamples": r[20], "VerificationFactor": VF,
                            "TrueNM": cm.loc[0, 0], "FalseM": cm.loc[0, 1],
                            "FalseNM": cm.loc[1, 0], "TrueM": cm.loc[1, 1]},
                           ignore_index=True)
    print("Time:", time.time() - time0)

    ########################################################
    #        Stochastic Gradient Boosting Classification

    ModelName = "StochasticGradientBoosting"

    print("************Stochastic Gradient Boosting Results**************")
    time0 = time.time()

    from sklearn.ensemble import GradientBoostingClassifier

    model1 = GradientBoostingClassifier(n_estimators=50, random_state=12).fit(X_train, Y_train)

    test_predictions = model1.predict(X_test)
    train_predictions = model1.predict(X_train)

    print("train results:")
    cm_train = confusion_matrix(Y_train, train_predictions)
    print(cm_train)

    cr_train = classification_report(Y_train, train_predictions)
    print(cr_train)
    print(accuracy_score(Y_train, train_predictions))

    print("test results:")
    cm_test = confusion_matrix(Y_test, test_predictions)
    print(cm_test)

    cr_test = classification_report(Y_test, test_predictions)
    print(cr_test)
    print(accuracy_score(Y_test, test_predictions))
    cm = pd.crosstab(test['RIO_ACCEPT_REJECT'], test_predictions, rownames=["Actual"], colnames=["predicted"])
    r = cr_test.split()
    VF = (cm.loc[0, 1] + cm.loc[1, 1]) / int(r[20])
    print("VF:  ", VF)
    result = result.append({"ModelName": ModelName, "Technique": Technique, "Class_Weight": classweight,
                            "NM_Precision": r[5], "NM_Recall": r[6],
                            "M_Precision": r[10], "M_Recall": r[11],
                            "NM_Samples": r[8], "M_Samples": r[13],
                            "TotalSamples": r[20], "VerificationFactor": VF,
                            "TrueNM": cm.loc[0, 0], "FalseM": cm.loc[0, 1],
                            "FalseNM": cm.loc[1, 0], "TrueM": cm.loc[1, 1]},
                           ignore_index=True)
    print("Time:", time.time() - time0)
    ###################################################
    ModelName = "BaggingClassifierDT"

    print("*************************Bagging Classifier DT***************")
    time0 = time.time()

    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier()
    model1 = BaggingClassifier(base_estimator=model, n_estimators=100, random_state=0).fit(X_train, Y_train)
    test_predictions = model1.predict(X_test)
    train_predictions = model1.predict(X_train)

    print("train results:")
    cm_train = confusion_matrix(Y_train, train_predictions)
    print(cm_train)

    cr_train = classification_report(Y_train, train_predictions)
    print(cr_train)
    print(accuracy_score(Y_train, train_predictions))

    print("test results:")
    cm_test = confusion_matrix(Y_test, test_predictions)
    print(cm_test)

    cr_test = classification_report(Y_test, test_predictions)
    print(cr_test)
    print(accuracy_score(Y_test, test_predictions))
    cm = pd.crosstab(test['RIO_ACCEPT_REJECT'], test_predictions, rownames=["Actual"], colnames=["predicted"])
    r = cr_test.split()
    VF = (cm.loc[0, 1] + cm.loc[1, 1]) / int(r[20])
    print("VF:  ", VF)
    result = result.append({"ModelName": ModelName, "Technique": Technique, "Class_Weight": classweight,
                            "NM_Precision": r[5], "NM_Recall": r[6],
                            "M_Precision": r[10], "M_Recall": r[11],
                            "NM_Samples": r[8], "M_Samples": r[13],
                            "TotalSamples": r[20], "VerificationFactor": VF,
                            "TrueNM": cm.loc[0, 0], "FalseM": cm.loc[0, 1],
                            "FalseNM": cm.loc[1, 0], "TrueM": cm.loc[1, 1]},
                           ignore_index=True)

    print("Time:", time.time() - time0)
'''

'''
##################################################
# SVM LINEAR
# --------------------------------
ModelName = "LinearSVM"

print("********************SVM linear Results***************")
time0 = time.time()

from sklearn import svm

model1 = svm.SVC(kernel='linear', C=1, class_weight=classweight).fit(X_train, Y_train)
test_predictions = model1.predict(X_test)
train_predictions = model1.predict(X_train)

print("train results:")
cm_train = confusion_matrix(Y_train, train_predictions)
print(cm_train)

cr_train = classification_report(Y_train, train_predictions)
print(cr_train)
print(accuracy_score(Y_train, train_predictions))

print("test results:")
cm_test = confusion_matrix(Y_test, test_predictions)
print(cm_test)

cr_test = classification_report(Y_test, test_predictions)
print(cr_test)
print(accuracy_score(Y_test, test_predictions))
cm = pd.crosstab(test['RIO_ACCEPT_REJECT'], test_predictions, rownames=["Actual"], colnames=["predicted"])
r = cr_test.split()
VF = (cm.loc[0, 1] + cm.loc[1, 1]) / int(r[20])
print("VF:  ", VF)
result = result.append({"ModelName": ModelName, "Technique": Technique, "Class_Weight": classweight,
                        "NM_Precision": r[5], "NM_Recall": r[6],
                        "M_Precision": r[10], "M_Recall": r[11],
                        "NM_Samples": r[8], "M_Samples": r[13],
                        "TotalSamples": r[20], "VerificationFactor": VF,
                        "TrueNM": cm.loc[0, 0], "FalseM": cm.loc[0, 1],
                        "FalseNM": cm.loc[1, 0], "TrueM": cm.loc[1, 1]},
                       ignore_index=True)
print("Time:", time.time() - time0)

##################################################
# SVM RADIAL
# --------------------------------
ModelName = "RadialSVM"

print("********************SVM Radial***************")
time0 = time.time()

from sklearn import svm

model1 = svm.SVC(kernel='rbf', gamma=0.7, C=1, class_weight=classweight).fit(X_train, Y_train)
test_predictions = model1.predict(X_test)
train_predictions = model1.predict(X_train)

print("train results:")
cm_train = confusion_matrix(Y_train, train_predictions)
print(cm_train)

cr_train = classification_report(Y_train, train_predictions)
print(cr_train)
print(accuracy_score(Y_train, train_predictions))

print("test results:")
cm_test = confusion_matrix(Y_test, test_predictions)
print(cm_test)

cr_test = classification_report(Y_test, test_predictions)
print(cr_test)
print(accuracy_score(Y_test, test_predictions))
cm = pd.crosstab(test['RIO_ACCEPT_REJECT'], test_predictions, rownames=["Actual"], colnames=["predicted"])
r = cr_test.split()
VF = (cm.loc[0, 1] + cm.loc[1, 1]) / int(r[20])
print("VF:  ", VF)
result = result.append({"ModelName": ModelName, "Technique": Technique, "Class_Weight": classweight,
                        "NM_Precision": r[5], "NM_Recall": r[6],
                        "M_Precision": r[10], "M_Recall": r[11],
                        "NM_Samples": r[8], "M_Samples": r[13],
                        "TotalSamples": r[20], "VerificationFactor": VF,
                        "TrueNM": cm.loc[0, 0], "FalseM": cm.loc[0, 1],
                        "FalseNM": cm.loc[1, 0], "TrueM": cm.loc[1, 1]},
                       ignore_index=True)
print("Time:", time.time() - time0)
'''


'''
####################################
#       KNN
#---------------------------------

ModelName="K-NearestNeighbors"


print("****************KNN Results******************")
time0=time.time()

from sklearn.neighbors import KNeighborsClassifier

k_range=range(1,100,2)
rec=[]

for k in k_range:
    print(k)
    model1=KNeighborsClassifier(n_neighbors=k).fit(X_train, Y_train)
    test_predictions = model1.predict(X_test)
    rec.append(metrics.f1_score(Y_test, test_predictions))
    print("precision:",metrics.f1_score(Y_test, test_predictions))

print("rec:",rec)

i=rec.index(max(rec))

k=k_range[i]

print("kmax:",k)
model1=KNeighborsClassifier(n_neighbors=k).fit(X_train, Y_train)
test_predictions = model1.predict(X_test)
train_predictions = model1.predict(X_train)

print("train results:")
cm_train = confusion_matrix(Y_train, train_predictions)
print(cm_train)

cr_train = classification_report(Y_train, train_predictions)
print(cr_train)
print(accuracy_score(Y_train, train_predictions))

print("test results:")
cm_test = confusion_matrix(Y_test, test_predictions)
print(cm_test)

cr_test = classification_report(Y_test, test_predictions)
print(cr_test)
print(accuracy_score(Y_test, test_predictions))
cm = pd.crosstab(test['RIO_ACCEPT_REJECT'], test_predictions, rownames=["Actual"], colnames=["predicted"])
r=cr_test.split()
VF=(cm.loc[0,1]+cm.loc[1,1])/int(r[20])
print("VF:  ",VF)
result=result.append({"ModelName":ModelName, "Technique":Technique,"Class_Weight":classweight,
                      "NM_Precision":r[5], "NM_Recall":r[6],
                      "M_Precision":r[10], "M_Recall":r[11],
                      "NM_Samples":r[8], "M_Samples":r[13],
                      "TotalSamples":r[20],"VerificationFactor":VF,
                      "TrueNM":cm.loc[0,0],"FalseM":cm.loc[0,1],
                      "FalseNM":cm.loc[1,0],"TrueM":cm.loc[1,1]},
    ignore_index=True)

print("Time:",time.time()-time0)

'''

print(result)

#result.to_csv("/home/mythri/PycharmProjects/BinaryClassification/Results/TestResults"+Technique+".csv",index=False)import pandas as pd
