#2025.3.10.
#프로젝트2 붓꽃분류기 만들기
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

iris_df= pd.read_csv('iris.csv')
print(iris_df)
y=iris_df['species']
X= iris_df.drop('species',axis=1)

print(y)
print(X)

kn=KNeighborsClassifier()
model_kn= kn.fit(X,y)

#X_new=np.array([5.0,3.4,3.5,1.4])
# X_new=np.array([5.0,3.4,3.5,1.4])
X_new=np.array([1,4.2,1.4,7])
prediction = model_kn.predict((X_new))
print(prediction)
probability = model_kn.predict_proba(X_new)
print(probability)