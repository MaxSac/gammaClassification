import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fact.analysis import li_ma_significance, split_on_off_source_independent

'''
With this function there are possibilities to get more information over the 
significance on the dataset by given estimators. 

There is a function which can calculated the highest significance. Another 
function which plot the significance in dependence of threshold and an on 
off ratio plotter.
'''
def load_feature():
		with open('/home/msackel/Desktop/gammaClassification/config/feature.yaml') as f:
			feature = yaml.load(f)
		return feature

def model_significance(estimator, data):
		'''
		Evaluate significance on given trained model and given datset.
		Parameters:
			estimator: sklearn.model
				Trained model, so there the estimator can make predictions 
				on the dataset.
			data: pd.DataFrame
				The dataset where the siginificance should be calculated
		Returns:
			max(significance): float
				Maximal signigicance on the dataset by given model.
		'''
		feature = load_feature()
		data['gamma_prediction'] = estimator.predict_proba(data[feature])[:,1]
		significance = []
		for threshold in np.linspace(0.01, 0.99, 99):
			on_data, off_data = split_on_off_source_independent(
							data.query('gamma_prediction >'+threshold.astype(str)),
							theta2_cut=0.03)
			significance.append(li_ma_significance(len(on_data), len(off_data), 0.2))
		return max(significance)

def plot_significance(estimator, data, save=True, path= 'significance.pdf'):
		'''
		Plot the significance in dependence to threshold.
		Parameters:
			estimator: sklearn.model
				Trained model, so there the estimator can make predictions 
				on the dataset.
			data: pd.DataFrame
				The dataset where the siginificance should be calculated
		'''
		feature = load_feature()
		data['gamma_prediction'] = estimator.predict_proba(data[feature])[:,1]
		significance = []
		for threshold in np.linspace(0.01, 0.99, 99):
			on_data, off_data = split_on_off_source_independent(
							data.query('gamma_prediction >'+threshold.astype(str)), 
							theta2_cut=0.03)
			significance.append(li_ma_significance(len(on_data), len(off_data), 0.2))
		plt.plot(np.linspace(0.01, 0.99, 99), significance)
		if(save==True):
			plt.title('max('+str(round(max(significance),2))+')')
			plt.xlabel('threshold')
			plt.ylabel('confidence')
			plt.savefig(path)

def plot_on_off_ratio(estimator, data, threshold, Bins=100, Range= [0,3], path='on_off_ratio.pdf'):
		'''
		Plot the classified signal and background in a histogramm. On the 
		x-axis is theta**2 on the other the number of events.
		Parameters:
			estimator: sklearn.model
				Trained model, so there the estimator can make predictions 
				on the dataset.
			data: pd.DataFrame
				The dataset where the siginificance should be calculated
		'''
		feature = load_feature()
		data['gamma_prediction'] = estimator.predict_proba(data[feature])[:,1]
		data['gamma_prediction'] = estimator.predict_proba(data[feature])[:,1]
		selected = data.query('gamma_prediction >= '+ str(threshold))
		theta_on = selected.theta_deg
		theta_off = pd.concat([selected['theta_deg_off_' + str(i)] for i in range(1, 6)])
		plt.hist(theta_on**2, range=[0, 0.3], bins=100, histtype='step', label='On')
		plt.hist(theta_off**2, range=[0, 0.3], bins=100, histtype='step', label='Off', 
						weights=np.full(len(theta_off),  0.2))

		plt.xlabel(r'theta$^{2}$')
		plt.ylabel(r'events')
		plt.savefig(path)
