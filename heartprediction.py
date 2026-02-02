import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('heart.csv')
print(df.head(20))
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
numeric_columns=['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']

ch_mean=df.loc[df['Cholesterol']!=0,'Cholesterol'].mean()  
df['Cholesterol']=df['Cholesterol'].replace(0,ch_mean)
bp_mean=df.loc[df['RestingBP']!=0,'RestingBP'].mean()
df['RestingBP']=df['RestingBP'].replace(0,bp_mean)

categories=['ExerciseAngina','ST_Slope','ChestPainType','Sex','RestingECG']
for i in categories:
    print(df[i].value_counts())

df['Sex']=df['Sex'].map(
    {
        'M':0,
        'F':1
    }
)
df['ExerciseAngina']=df['ExerciseAngina'].map({
    'Y':1,
    'N':0
})

df=pd.get_dummies(df,columns=['ChestPainType','RestingECG','ST_Slope'])
print(df.head())
df=df.astype(int)
x_normal=df.drop('HeartDisease',axis=1)

scale=StandardScaler()
df[numeric_columns]=scale.fit_transform(df[numeric_columns])

X=df.drop('HeartDisease',axis=1)
y=df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


model_lr=LogisticRegression()
model_lr.fit(X_train,y_train)
y_pred=model_lr.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(f1_score(y_test,y_pred))

model_svm=SVC()
model_svm.fit(X_train,y_train)
y_pred=model_svm.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(f1_score(y_test,y_pred),"\n")

model_naive=GaussianNB()
model_naive.fit(X_train,y_train)
y_pred=model_naive.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(f1_score(y_test,y_pred),"\n")

model_dt=DecisionTreeClassifier()
model_dt.fit(X_train,y_train)
y_pred=model_dt.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(f1_score(y_test,y_pred),"\n")

model_KNN=KNeighborsClassifier()
model_KNN.fit(X_train,y_train)
y_pred=model_KNN.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(f1_score(y_test,y_pred),"\n")

joblib.dump(model_lr,"lr_heart.pkl")
joblib.dump(scale,"scaler.pkl")
columns = X.columns.tolist()

import json
with open("columns.json", "w") as f:
    json.dump(columns, f)

print("Correct columns.json saved")






