import numpy as np
import pandas as pd
from sklearn import linear_model,datasets
from sklearn.externals import joblib 
from sklearn.model_selection import train_test_split
iris=datasets.load_iris()
df=pd.DataFrame(iris.data)
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(df,y,test_size=0.2)
model=linear_model.LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
# Save the model as a pickle in a file 
print(y_pred)
print(y_test)
joblib.dump(model, 'filename.pkl') 
print(y_pred)
