import pandas as pd
import numpy as np
data = pd.read_csv('titanic.csv')
data.replace('?', np.nan, inplace=True)
data = data.astype({"age": np.float64, "fare": np.float64})
data.replace({'male': 1, 'female': 0}, inplace=True)

data = data[['sex', 'pclass','age','fare','survived']].dropna()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data[['sex','pclass',"age",'fare']], data.survived, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)

from sklearn import metrics
predict_test = model.predict(X_test)
print(metrics.accuracy_score(y_test, predict_test))