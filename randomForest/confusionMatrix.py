import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
plt.style.use('ggplot')

dataCrab = pd.read_pickle('../data/dataCrab')
dataSimu = pd.read_pickle('../data/dataSimu')

X_train, X_test, y_train, y_test = train_test_split(dataCrab.drop('label', 
	axis=1),dataCrab.label, random_state=0)

Tree = RandomForestClassifier(random_state=0, max_depth=13, n_estimators=500,
	n_jobs=18,verbose=1)
Tree.fit(X_train, y_train)
y_pred = Tree.predict(X_test)

cnf_matrix = confusion_matrix(y_test, y_pred)
plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.xticks([0,1],['gamma','hadron'], rotation=45)
plt.yticks([0,1],['gamma','hadron'])
plt.colorbar()

cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
thresh = cnf_matrix.max() / 2.

for i, j in itertools.product(range(cnf_matrix.shape[0]), 
		range(cnf_matrix.shape[1])):
	plt.text(j, i, np.round(cnf_matrix[i, j], decimals=3), 
		horizontalalignment="center", 
		color="white" if cnf_matrix[i, j] > thresh else "black",
		fontsize=20)

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.savefig('../plots/confusionMatrix.pdf')
