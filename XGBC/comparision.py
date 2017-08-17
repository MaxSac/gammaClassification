import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from matplotlib import pyplot as plt
plt.style.use('msackel')

data= pd.read_pickle('/home/msackel/Desktop/gammaClassification/data/dataSimu')

parameters = {
				'nthread':20,
				'learning_rate':0.1,
				'objective':'binary:logistic',         
				'max_depth':5,
				'gamma':1,
				'min_child_weight':0.5,
				'subsample':0.3,
				'colsample_bytree':.7,
				'scale_pos_weight':1,
				}
rnd = XGBClassifier(**parameters)
rnd.fit(data.drop('label', axis=1), data.label)

pre = rnd.predict_proba(data.drop('label', axis=1))[:,1]

Tree = RandomForestClassifier(n_estimators=100, max_depth=15, max_features=6, n_jobs=30, verbose=1, bootstrap=True, criterion='entropy')
Tree.fit(data.drop('label', axis=1), data.label)
Pre = Tree.predict_proba(data.drop('label', axis=1))[:,1]

print(cross_val_score(Tree, data.drop('label', axis=1), data.label).mean())
print(cross_val_score(rnd, data.drop('label', axis=1), data.label).mean())

SC_T = cross_val_score(Tree, data.drop('label', axis=1), data.label)
SC_X = cross_val_score(rnd, data.drop('label', axis=1), data.label)

plt.hist(pre, bins=50, range=[0,1], histtype='step', label='XGBC: '+str(np.round(SC_T.mean(), 3))+' +- '+ str(np.round(SC_T.std(), 3)))
plt.hist(Pre, bins=50, range=[0,1], histtype='step', label='Tree: '+str(np.round(SC_X.mean(), 3))+' +- '+ str(np.round(SC_X.std(), 3)))
plt.yscale('log')
plt.ylim(1e3, 4e4)
plt.legend()
plt.savefig('compare.pdf')
