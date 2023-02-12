import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\Rajesh Yadav\Desktop\Crop_Recommendation\Crop_recommendation.csv")
X = df.drop(['label'], axis=1)
y = df.label

# data preprocessing
from sklearn.preprocessing import StandardScaler

X_standardized = X.copy()
for col in X_standardized.columns:
    X_standardized[col] = StandardScaler().fit_transform(X_standardized[col].values.reshape(-1, 1))
    
X_standardized.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, train_size=0.8, random_state=42)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit(X_train, y_train)
#y_pred = model.predict(X_test)

import joblib
joblib.dump(model, 'model.pkl')