import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fact.analysis import li_ma_significance, split_on_off_source_independent
plt.style.use('ggplot')

'''
params: fitted estimator
'''

def model_significance(estimator, data):
		data['gamma_prediction'] = Tree.predict_proba(data[feature])[:,1]
		significance = []
		for threshold in np.linspace(0.01, 0.99, 99):
			on_data, off_data = split_on_off_source_independent(hadron_data.query('gamma_prediction >'+threshold.astype(str)), theta2_cut=0.03)
			significance.append(li_ma_significance(len(on_data), len(off_data), 0.2))
		return max(significance)

def plot_significance(estimator, data):
		data['gamma_prediction'] = Tree.predict_proba(data[feature])[:,1]
		significance = []
		for threshold in np.linspace(0.01, 0.99, 99):
			on_data, off_data = split_on_off_source_independent(hadron_data.query('gamma_prediction >'+threshold.astype(str)), theta2_cut=0.03)
			significance.append(li_ma_significance(len(on_data), len(off_data), 0.2))
		plt.plot(np.linspace(0.01, 0.99, 99), significance)

		plt.xlabel('threshold')
		plt.ylabel('confidence')
		plt.savefig('significance.pdf')

def plot_on_off_ratio(estimator, data, threshold, Bins=100, Range= [0,3]):
		data['gamma_prediction'] = Tree.predict_proba(data[feature])[:,1]
		on_data, off_data = split_on_off_source_independent(hadron_data.query('gamma_prediction >'+threshold), theta2_cut=0.03)
		plt.hist(on_data, histtype='step', bins=Bins, range=Range)


from fact.io import read_h5py
from sklearn.ensemble import RandomForestClassifier
feature = ['conc_core',
				'concentration_one_pixel',    
				'concentration_two_pixel',
				'leakage', 
				'leakage2', 
				'size', 
				'width', 
				'num_islands', 
				'num_pixel_in_shower', 
				'ph_charge_shower_max',
				'ph_charge_shower_mean',
				'ph_charge_shower_min',
				'ph_charge_shower_variance']

hadron_data = read_h5py(
				'/home/msackel/Desktop/gammaClassification/data/raw_data/crab_precuts.hdf5',
				key='events',
				columns=list(feature)+ [
						'theta_deg',
						'theta_deg_off_1',
						'theta_deg_off_2',
						'theta_deg_off_3',
						'theta_deg_off_4',
						'theta_deg_off_5',
						]
				)

crab = pd.read_pickle('/home/msackel/Desktop/gammaClassification/data/complete_Datasets/dataCrab')

Tree = RandomForestClassifier(n_estimators=100, n_jobs=25).fit(crab.drop('label', axis=1), crab.label)
#print('max significance: ', model_significance(Tree, hadron_data))
#plot_significance(Tree, hadron_data)
plot_on_off_ratio(Tree, hadron_data, '0.8')
