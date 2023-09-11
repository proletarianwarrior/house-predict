# -*- coding: utf-8 -*-
# @Time : 2022/9/21 21:02
# @Author : DanYang
# @File : main.py
# @Software : PyCharm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import POI
import numpy as np

data = POI.poi()
data = np.array(data)
X, y = data[[True, True, True, True, True, True, True, False, False, True]], data[-3]

X_train, X_test, y_train, y_test = train_test_split(X.T, y.astype(int), random_state=0)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
precision = clf.score(X_test, y_test)
print(precision)
