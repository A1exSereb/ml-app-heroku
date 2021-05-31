import pandas as pd
from sklearn.linear_model import LogisticRegression
students = pd.read_csv('diplom-filtered.csv')


df = students.copy()
target = 'enrolled'



# Разделение X и Y
X = df.drop('enrolled', axis=1)
Y = df['enrolled']

# Построение random forest модели
clf = LogisticRegression()
clf.fit(X, Y)

# Сохраняет модель
import pickle
pickle.dump(clf, open('college_model.pkl', 'wb'))