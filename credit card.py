#importing necessary  liabraries
import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score
from imblearn.over_sampling import SMOTE

#load the dataset
df=pd.read_csv("creditcard.csv")
print(df.head(10))

#scaling the  "Amount" and "Time" features by StandardScaler
df["Amount"]=StandardScaler().fit_transform(df["Amount"].values.reshape(-1,1))
df["Time"]=StandardScaler().fit_transform(df["Time"].values.reshape(-1,1))
print(df["Amount"].value_counts())
print(df["Time"].value_counts())

#seperate the features(x) and values(y)
x=df.drop("Class",axis=1)
y=df["Class"]

#spliting the values in training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

#apply SMOTE to balance the trainig data
x_sm,y_sm=SMOTE(random_state=42).fit_resample(x_train,y_train)
print(y_sm.value_counts())

#initialize and train RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train)

#make predictions on test data
y_pred=model.predict(x_test)
y_prob=model.predict_proba(x_test)[:,1]

#Evaluate performance by using Confusion matrix and Classification Report
cm=confusion_matrix(y_test,y_pred)
print("Confusion matrix:",cm)
cr=classification_report(y_test,y_pred)
print("Classification Report:",cr)

#Calculate and display Roc Auc Score
score=roc_auc_score(y_test,y_prob)
print("Roc Auc Score:",score)

#Visualize Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.legend("Confusion Matrix")
plt.show()

