import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


df = pd.read_csv('C:/Users/anany/Downloads/survey lung cancer 1 - Copy.csv')
X = df[['GENDER','AGE','SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC DISEASE','FATIGUE ','ALLERGY ','WHEEZING','ALCOHOL CONSUMING','COUGHING','SHORTNESS OF BREATH','SWALLOWING DIFFICULTY','CHEST PAIN']]
y = df['LUNG_CANCER']
x1=df['AGE']
y1 = pd.cut(x = df['LUNG_CANCER'], bins = [0,1,2],labels=["Cancer","No Cancer"])
plt.xlabel("Age")
plt.bar(x1,y1)
plt.show()

#yes is 1 no is 2
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)


"""
neigh = [1,3,5,7,9,11,13,15] 

Testing for best K-val (k=7)

for x in neigh:
 KNN_ex = KNeighborsClassifier(n_neighbors=x)
 KNN_ex.fit(X_train,y_train)
 y_pred = KNN_ex.predict(X_test)
 print(accuracy_score(y_test,y_pred))"""

KNN_ex = KNeighborsClassifier(n_neighbors=7)
KNN_ex.fit(X_train,y_train)
y_pred = KNN_ex.predict(X_test)
print(accuracy_score(y_test,y_pred))

cdisplay = confusion_matrix(y_pred = y_pred, y_true = y_test)
c1display = metrics.ConfusionMatrixDisplay(confusion_matrix=cdisplay)
c1display.plot()
plt.show()