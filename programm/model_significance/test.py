import pandas as pd
import numpy as np
from fact.analysis import li_ma_significance, split_on_off_source_independent

'''
params: fitted estimator
'''

def model_significance(estimator, data):
		data['gamma_prediction'] = Tree.predict_proba(data[feature])[:,1]
		significance = []
		for threshold in np.linspace(0.01, 0.99, 99):
			on_data, off_data = split_on_off_source_independent(hadron_data.query('gamma_prediction >= 0.5'), theta2_cut=0.03)
			#significance.append(li_ma_significance(len(on_data), len(off_data), 0.2))
		return max(significance)

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

Tree = RandomForestClassifier().fit(crab.drop('label', axis=1), crab.label)
model_significance(Tree, hadron_data)
