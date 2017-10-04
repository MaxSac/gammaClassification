import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
plt.style.use('msackel')

with open('/home/msackel/Desktop/gammaClassification/config/feature.yaml') as f:
	feature = yaml.load(f)
feature.append('theta_deg')

gammas = pd.read_hdf('/home/msackel/Desktop/gammaClassification/data/raw_data/gamma_precuts.hdf5')[feature]
proton = pd.read_hdf('/home/msackel/Desktop/gammaClassification/data/raw_data/proton_precuts.hdf5')[feature]

gammas['label'] = 1
proton['label'] = 0

data = pd.concat([gammas, proton])

X_train, X_test, y_train, y_test = train_test_split(data.drop('label', axis=1), data.label)

Tree_ohne_theta = RandomForestClassifier(
				    n_estimators=100, 
					    n_jobs=15, 
						    max_depth=15, 
							    bootstrap=True)

Tree_mit_theta = RandomForestClassifier(
				    n_estimators=100, 
					    n_jobs=15, 
						    max_depth=15, 
							    bootstrap=True)

Tree_ohne_theta.fit(X_train.drop('theta_deg', axis=1), y_train)
Tree_mit_theta.fit(X_train, y_train)

pred = Tree_ohne_theta.predict_proba(X_test.drop('theta_deg', axis=1))[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, pred)

Pred = Tree_mit_theta.predict_proba(X_test)[:,1]
Fpr, Tpr, Threshold = metrics.roc_curve(y_test, Pred)

fig = plt.figure(figsize=(2.728,2.1824))
plt.plot(fpr,tpr, label=r'ohne $\theta$')
plt.plot(Fpr,Tpr, label=r'mit $\theta$')
plt.plot(np.linspace(0,1,10), np.linspace(0,1,10), '-.', label=r'Zufall')
plt.legend()
plt.xlabel('false positiv rate')
plt.ylabel('true positiv rate')

plt.savefig('plots/roc_info.pdf')
