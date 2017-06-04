import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
plt.style.use('ggplot')

dataCrab = pd.read_pickle('../data/dataCrab')
dataSimu = pd.read_pickle('../data/dataSimu')

X_train, X_test, y_train, y_test = train_test_split(dataCrab.drop('label', 
	axis=1),dataCrab.label, random_state=0)

accuracy = np.array([[],[],[]])
for n in np.linspace(1,20,15):
	Tree = RandomForestClassifier(random_state=0, max_depth=n, 
			n_estimators=200, n_jobs=18,verbose=1)
	Tree.fit(X_train, y_train)
	accuracy = np.append(accuracy, [[n],[Tree.score(X_train, y_train)],
		[Tree.score(X_test, y_test)]], axis=1)

plt.title('Training and Test score for different max_depth')
plt.plot(accuracy[0], accuracy[1], label=r'training-set')
plt.plot(accuracy[0], accuracy[2], label=r'test-set')
plt.xlabel('depth')
plt.ylabel('score')
plt.legend(loc='best')
plt.savefig('../plots/bestMax_depth.pdf')
