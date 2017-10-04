import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from fact.analysis import li_ma_significance, split_on_off_source_independent
exec(open('/home/msackel/Desktop/gammaClassification/programm/theta_cut/theta_cut.py').read())
plt.style.use('msackel')

with open('/home/msackel/Desktop/gammaClassification/config/feature.yaml') as f:
		feature = yaml.load(f)

mess_data = theta_cut('/home/msackel/Desktop/gammaClassification/data/raw_data/gamma_precuts.hdf5', 
				'/home/msackel/Desktop/gammaClassification/data/raw_data/crab_precuts.hdf5', 0.5, length=100000) 
mc_data = pd.read_pickle('/home/msackel/Desktop/gammaClassification/data/dataSimu')

eval_data = read_h5py(
				'/home/msackel/Desktop/gammaClassification/data/raw_data/crab_precuts.hdf5',
				key='events',
				columns=list(feature) + [
						'theta_deg',
						'theta_deg_off_1',
						'theta_deg_off_2',
						'theta_deg_off_3',
						'theta_deg_off_4',
						'theta_deg_off_5']
				)


mess_Tree = RandomForestClassifier(max_depth=15, max_features=7, criterion='entropy', n_estimators=100, n_jobs=20)
mc_Tree = RandomForestClassifier(max_depth=15, max_features=7, criterion='entropy', n_estimators=100, n_jobs=20)
mess_xgbc = XGBClassifier(max_depth= 1, learning_rate= 0.5, n_estimators= 100, booster= 'gbtree', gamma= 0.5, n_jobs= 20, reg_lambda= 0.1, subsample= 0.20, eval_metric= 'auc')
mc_xgbc = XGBClassifier(max_depth= 1, learning_rate= 0.5, n_estimators= 100, booster= 'gbtree', gamma= 0.5, n_jobs= 20, reg_lambda= 0.1, subsample= 0.20, eval_metric= 'auc')


mess_Tree.fit(mess_data.drop('label', axis=1),mess_data.label)
mess_xgbc.fit(mess_data.drop('label', axis=1),mess_data.label)
mc_Tree.fit(mc_data.drop('label', axis=1),mc_data.label)
mc_xgbc.fit(mc_data.drop('label', axis=1),mc_data.label)


pred_mess_tree = mess_Tree.predict_proba(eval_data[feature])[:,1]
pred_mess_xgbc = mess_xgbc.predict_proba(eval_data[feature])[:,1]
pred_mc_tree = mc_Tree.predict_proba(eval_data[feature])[:,1]
pred_mc_xgbc = mc_xgbc.predict_proba(eval_data[feature])[:,1]


sig_mess_tree = []
sig_mess_xgbc= []
sig_mc_tree = []
sig_mc_xgbc= []
for threshold in np.linspace(0.01, 0.99, 99):
		on_data, off_data = split_on_off_source_independent(eval_data[threshold <= pred_mess_tree],theta2_cut=0.03)
		sig_mess_tree.append(li_ma_significance(len(on_data), len(off_data), 0.2))
		on_data, off_data = split_on_off_source_independent(eval_data[threshold <= pred_mess_xgbc],theta2_cut=0.03)
		sig_mess_xgbc.append(li_ma_significance(len(on_data), len(off_data), 0.2))
		on_data, off_data = split_on_off_source_independent(eval_data[threshold <= pred_mc_tree],theta2_cut=0.03)
		sig_mc_tree.append(li_ma_significance(len(on_data), len(off_data), 0.2))
		on_data, off_data = split_on_off_source_independent(eval_data[threshold <= pred_mc_xgbc],theta2_cut=0.03)
		sig_mc_xgbc.append(li_ma_significance(len(on_data), len(off_data), 0.2))

data = pd.DataFrame({'sig_mess_tree':np.transpose(sig_mess_tree), 
		'sig_mess_xgbc':np.transpose(sig_mess_xgbc),
		'sig_mc_tree':np.transpose(sig_mc_tree), 
		'sig_mc_xgbc':np.transpose(sig_mc_xgbc)})
data.to_pickle('storage/conf_sign_data')

fig = plt.figure(figsize=(2.728,2.046))
plt.plot(np.linspace(0.01, 0.99, 99),sig_mess_xgbc, label='Mess')
plt.plot(np.linspace(0.01, 0.99, 99),sig_mc_xgbc, label='MC')
plt.ylabel('maximale Signifikanz')
plt.xlabel('Konfidenz-Schnitt')
plt.legend()
plt.ylim(20,42)
plt.savefig('plots/sig_mess_xgbc.pdf')
plt.close(fig)

Fig = plt.figure(figsize=(2.728,2.046))
plt.plot(np.linspace(0.01, 0.99, 99),sig_mess_tree, label='Mess')
plt.plot(np.linspace(0.01, 0.99, 99),sig_mc_tree, label='MC')
plt.ylabel('maximale Signifikanz')
plt.xlabel('Konfidenz-Schnitt')
plt.legend()
plt.ylim(20,42)
plt.savefig('plots/sig_mess_tree.pdf')
plt.close(Fig)
