import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from sklearn.ensemble import RandomForestClassifier
exec(open('/home/msackel/Desktop/gammaClassification/programm/theta_cut/theta_cut.py').read())
plt.style.use('msackel')

with open('/home/msackel/Desktop/gammaClassification/config/feature.yaml') as f:
	feature = yaml.load(f)


mess_data = theta_cut('/home/msackel/Desktop/gammaClassification/data/raw_data/gamma_precuts.hdf5', 
					'/home/msackel/Desktop/gammaClassification/data/raw_data/crab_precuts.hdf5', 0.5, length=100000) 
mc_data = pd.read_pickle('/home/msackel/Desktop/gammaClassification/data/dataSimu')

eval_data = read_h5py('/home/msackel/Desktop/gammaClassification/data/raw_data/crab_precuts.hdf5',
	key='events',
	columns=list(feature)
	)

mess_Tree = RandomForestClassifier(max_depth=15, max_features=7, criterion='entropy', n_estimators=100, n_jobs=30)
mc_Tree = RandomForestClassifier(max_depth=15, max_features=7, criterion='entropy', n_estimators=100, n_jobs=30)

mess_Tree.fit(mess_data.drop('label', axis=1), mess_data.label)
mc_Tree.fit(mc_data.drop('label', axis=1), mc_data.label)

mc_pred = mc_Tree.predict_proba(eval_data)[:,1]
mess_pred = mess_Tree.predict_proba(eval_data)[:,1]

plt.figure(figsize=(5,3.75))
plt.hist(mc_pred, bins=20, histtype='step', label='MC-Daten')
plt.hist(mess_pred, bins=20, histtype='step', label='Messdaten')
plt.xlabel(r'Konfidenz')
plt.legend()
plt.savefig('plots/conf.pdf')
