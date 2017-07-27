'''
Read a gamma and hadron .hdf5 file and make theta cut on hadron set. Create an 
DataFrame with same parts of gammas and hadrons. 
'''
import pandas as pd
import yaml
from fact.io import read_h5py

def theta_cut(path_gamma, path_hadron, theta_cut, length=None,
				path_feature='/home/msackel/Desktop/gammaClassification/config/feature.yaml'):
	'''
	Read to concanate features from *.yaml file. And load the data from the given files.
	Hadron file added theta_deg feature to make theta cut on data.
	'''
	with open(path_feature) as f:
		feature = yaml.load(f)
	
	gamma_data = pd.read_hdf(path_gamma, key='events')[feature]
	hadron_data = read_h5py(path_hadron, key='events', columns=feature+ 
					['theta_deg'])

	'''
	Theta cut on hadron data and label data.
	'''
	hadron_data = hadron_data[hadron_data['theta_deg']**2 >= theta_cut]
	hadron_data['label'] = 0
	gamma_data['label'] = 1
	
	'''
	Length is set to the minimal lenght of the both data and a concat dataset is returned.
	'''
	if(length==None):
		length = min([len(hadron_data), len(gamma_data)])
	

	return pd.concat([hadron_data.drop('theta_deg', axis=1)[:length], gamma_data[:length]])
