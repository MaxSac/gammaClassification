import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
plt.style.use('ggplot')

dataCrab = pd.read_pickle('../data/dataCrab')

Tree = RandomForestClassifier(random_state=0, max_depth=13, n_estimators=500,
	n_jobs=18,verbose=1)
Tree.fit(dataCrab.drop('label', axis=1), dataCrab.label)

sequence = np.argsort(Tree.feature_importances_)

plt.barh(range(len(Tree.feature_importances_)), 
		Tree.feature_importances_[sequence] , align='center')
plt.yticks(range(len(Tree.feature_importances_)), 
		dataSimu.drop('label', axis=1).columns[sequence])
plt.xlabel(r'Feature importance')
plt.tight_layout()
plt.savefig('../plots/featureImportance.pdf')
