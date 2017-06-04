import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
plt.style.use('ggplot')

dataCrab = pd.read_pickle('../data/dataCrab')

XTrain, XTest, yTrain, yTest = train_test_split(dataCrab.drop('label', axis=1),
	dataCrab.label, random_state=42, test_size = 0.1)

for x in range(15):
	X_train, X_test, y_train, y_test = train_test_split(XTrain, yTrain, 
		random_state=42, test_size = 0.5)
	print('shape of XTrain:', XTrain.shape)
	print('shape of yTrain:', yTrain.shape)

	Tree = RandomForestClassifier(random_state=0, max_depth=16, 
		n_estimators=600, n_jobs=30)
	Tree.fit(X_train, y_train)
	print(x, 'Score: ', Tree.score(X_test,y_test))
	pred = Tree.predict(XTrain)
	print('Confusion Matrix: \n', confusion_matrix(yTrain, pred))
	XTrain ,yTrain = XTrain[pred == yTrain], yTrain[pred == yTrain]

print('Result of rekusive learining: ', Tree.score(XTest, yTest))
pred = Tree.predict(XTest)
print('Confusion Matrix: \n', confusion_matrix(yTest, pred))
selfMadeData = pd.concat([XTrain, yTrain], axis=1)
selfMadeData.to_pickle('../data/selfMadeData')
