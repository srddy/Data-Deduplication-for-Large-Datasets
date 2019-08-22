import pandas as pd
import numpy as np
import pickle

# 1. READ DATA TO BE PREPROCESSED

part1 = pd.read_csv("/Users/sandeepreddygopu/Desktop/Mini Project/data/ra_results_analysis_01_16jun2017.csv")
part2 = pd.read_csv("/Users/sandeepreddygopu/Desktop/Mini Project/data/ra_results_analysis_02_16jun2017.csv")
data = part1.append(part2)
#remove SM rows+
data = data[data['RIO_ACCEPT_REJECT'] != 'SM']

from sklearn.model_selection import train_test_split

#train, test = train_test_split(data, stratify=data.RIO_ACCEPT_REJECT.tolist(), test_size=0.3, random_state=13)

# Assume train and test are read from different files

###########Train Data##################
#print(train.isnull().sum())
data = data[[c for c  in list(data)  if len(data[c].unique()) > 1]]
datadropcols1=["SRCCOL11","TRGCOL11","TRGCOL18","REQUESTID","PSX_ID","SRCCOL28","TRGCOL28",
                "CACHETIMESTAMP","DUI_FLAG","PURGE_BATCH_ID","WEIGHTAGE","SCALE_TYPE"]
for c in data.columns:
    if c in datadropcols1:
        data.drop(c,inplace=True,axis=1)
data.dropna(inplace=True) # NAs removed

cat_columns=["RIO_RECORD_TYPE","RECORDTYPE","COL1000LVL","COL1040LVL","COL1050LVL"]
from sklearn.preprocessing import LabelEncoder
le_dict = {col: LabelEncoder() for col in cat_columns }
for col in cat_columns:
    le_dict[col].fit(data[col])
with open("/Users/sandeepreddygopu/Desktop/Mini Project/mini project.py/le_dict.pkl", 'wb') as file:  
    pickle.dump(le_dict, file)

for col in cat_columns:
    data[col]=le_dict[col].transform(data[col])

from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder(sparse=False)
for col in cat_columns:
       # creating an exhaustive list of all possible categorical values
       X=data[col].values.reshape(len(data[col]), 1)
       enc.fit(X)
       with open("/Users/sandeepreddygopu/Desktop/Mini Project/mini project.py/enc_"+col+".pkl", 'wb') as file:  
               pickle.dump(enc, file)
       # Fitting One Hot Encoding on train data
       temp = enc.transform(data[[col]])
       # Changing the encoded features into a data frame with new column names
       temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
            .value_counts().index])
       # In side by side concatenation index values should be same
       # Setting the index values similar to the X_train data frame
       temp=temp.set_index(data.index.values)
       # adding the new One Hot Encoded varibales to the train data frame
       data=pd.concat([data,temp],axis=1)
       # fitting One Hot Encoding on test data
       
data.drop(cat_columns,axis=1,inplace=True)

# 11. CLEAN MATCHTYPE FEATURE
data['MATCH_TYPE'] = data['MATCH_TYPE'].astype('str')
#print(data.groupby('RIO_ACCEPT_REJECT').count().MATCH_TYPE)
#data.info()
data['MATCH_TYPE']=data['MATCH_TYPE'].str.upper()
#clean MATCH_TYPE
data["MATCH_TYPE"] = data["MATCH_TYPE"].str.replace(pat=",", repl=" ")
#print(data["MATCH_TYPE"][0])

###removing extra whitespaces from text if any
f = lambda x: ' '.join(x["MATCH_TYPE"].split())
data['MATCH_TYPE'] = data.apply(f, axis=1)
#print(data["MATCH_TYPE"][0])

# 12. ADD MATCHCOUNT
g = lambda x: len(x["MATCH_TYPE"].split())
data['MATCH_COUNT'] = data.apply(g, axis=1)
#print(data.head())

'''
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud(collocations=False).generate(' '.join(data.MATCH_TYPE))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
'''

# 13. VECTORIZE MATCHTYPE
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#tfidf_vectorizer = TfidfVectorizer()
count_vectorizer=CountVectorizer()
#count_vectorizer=CountVectorizer(ngram_range=(1,3))
#tfidf = tfidf_vectorizer.fit_transform(train['MATCH_TYPE'])
VectorModelCV = count_vectorizer.fit(data['MATCH_TYPE'])

with open("/Users/sandeepreddygopu/Desktop/Mini Project/mini project.py/VectorModelCV.pkl", 'wb') as file:  
    pickle.dump(VectorModelCV, file)

countdf = VectorModelCV.transform(data['MATCH_TYPE'])

#print(tfidf.dtype)
dense = countdf.todense()
x = pd.DataFrame(dense)
x.columns = count_vectorizer.get_feature_names()
#print(x.index)
#print(x.count())
#print(dtrain.index)
# index problem resolution for concatanation
x.reset_index(drop=True, inplace=True)
data.reset_index(drop=True, inplace=True)

# 14.CONCATANATE VECTORIZED FEATURES WITH OTHER FEATURES
data=pd.concat([x,data],axis=1)
print(data.shape)

# 15. DROP REDUNDANT COLUMNS
#data.drop(["RIO_RECORD_TYPE","RECORDTYPE","SCALE_TYPE","MATCH_TYPE","COL1000LVL","COL1040LVL","COL1050LVL"],inplace=True,axis=1)
data.drop("MATCH_TYPE",inplace=True,axis=1)

# 16. SOME DATA MANIPULATIONS
data["COL1000BMCNT"].replace(100, -1, inplace=True)
data["COL1000BMCNT"].replace(-1, data["COL1000BMCNT"].max(), inplace=True)

data["COL1040BMCNT"].replace(100, -1, inplace=True)
data["COL1040BMCNT"].replace(-1, data["COL1040BMCNT"].max(), inplace=True)

data["COL1050BMCNT"].replace(100, -1, inplace=True)
data["COL1050BMCNT"].replace(-1, data["COL1050BMCNT"].max(), inplace=True)

data.drop(["CS1STRENGTH","CS2STRENGTH","CS3STRENGTH",
  "ADDR1MINLENSTRENGTH","ADDR2MINLENSTRENGTH","ADDR3MINLENSTRENGTH",
  "ADDR1MAXLENSTRENGTH","ADDR2MAXLENSTRENGTH","ADDR3MAXLENSTRENGTH"],inplace=True,axis=1)

#data.COL1050LVL.value_counts()
#data.iloc[:,40:].head()
#-1 in data.COL1000STRENGTH.unique()

data['flag1000'] = np.where(data['COL1000STRENGTH']==-1,1,0)
data['flag1040'] = np.where(data['COL1040STRENGTH']==-1,1,0)
data['flag1050'] = np.where(data['COL1050STRENGTH']==-1,1,0)
data['flag2000'] = np.where(data['COL2000STRENGTH']==-1,1,0)
data['flag3000'] = np.where(data['COL3000STRENGTH']==-1,1,0)
data['flag4000'] = np.where(data['COL4000STRENGTH']==-1,1,0)
#data['flag5000'] = np.where(data['COL5000STRENGTH']==-1,1,0)
#data['flag5010'] = np.where(data['COL5010STRENGTH']==-1,1,0)
#data['flag5180'] = np.where(data['COL5180STRENGTH']==-1,1,0)
data['flag5030'] = np.where(data['COL5030STRENGTH']==-1,1,0)
data['flag5040'] = np.where(data['COL5040STRENGTH']==-1,1,0)
print(data.shape)

# 9. PREPARE TARGET COLUMN
#data['RIO_ACCEPT_REJECT'] = data['RIO_ACCEPT_REJECT'].map({'NM': 0, 'M': 1,'SM':1})
data['RIO_ACCEPT_REJECT'] = data['RIO_ACCEPT_REJECT'].map({'NM': 0, 'M': 1,})
data['RIO_ACCEPT_REJECT'] = data['RIO_ACCEPT_REJECT'].astype('category')


# REMOVE UNIQUE COLUMNS
data = data[[c for c  in list(data)  if len(data[c].unique()) > 1]]
print(data.shape)

train, test = train_test_split(data, stratify=data.RIO_ACCEPT_REJECT.tolist(), test_size=0.3, random_state=13)

from sklearn.preprocessing import MinMaxScaler

X_train = train.loc[:, data.columns != 'RIO_ACCEPT_REJECT']
Y_train = train.RIO_ACCEPT_REJECT

scaler = MinMaxScaler().fit(X_train)
with open("/Users/sandeepreddygopu/Desktop/Mini Project/mini project.py/scaler.pkl", 'wb') as file:  
    pickle.dump(scaler, file)

#with open("/home/mythri/PycharmProjects/FinalClassification/SavedModels/scaler.pkl", 'rb') as file:  
#    scaler2=pickle.load(file)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)

#data.to_csv("/home/mythri/PycharmProjects/FinalClassification/Data/FullDataNoSM.csv",index=False)
print(X_train.head())
#train=pd.concat([X_train,Y_train],axis=1)
print(train.shape)


train.to_csv("/Users/sandeepreddygopu/Desktop/Mini Project/mini project.py/TrainData.csv",index=False)
test.to_csv("/Users/sandeepreddygopu/Desktop/Mini Project/mini project.py/TestData.csv",index=False)






