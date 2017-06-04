import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import h5py
from fact.io import read_h5py, read_h5py_chunked
import pickle
plt.style.use('ggplot')

"""This are the most relevant features,
for source indibendent classification. There is although the feature theta to 
make a theta cut because in the crab dataset are a few gammas too. Because 
theta is the best parameter for classification I make a litlle gamma cut to 
tread sure that there are only hadrons in the crab-set.
"""
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
	'ph_charge_shower_variance',
	'theta_deg']

"""
Read the data from a hdf5 file and import as, as a Pandas.DataFrame
"""
print('-----loading proton and gamma datas')
proton = pd.read_hdf('data/proton_precuts.hdf5')
gammas = pd.read_hdf('data/gamma_precuts.hdf5')

"""
Write all features from the crab dataset sorted after the first letters.
"""
with h5py.File('data/crab_precuts.hdf5', 'r') as f:
	    print(*sorted(f['events'].keys(), key=str.lower), sep='\n')

"""
For a lower loadingtime not the whole dataset is loading. Whit the command next
the next block is loading in the same dimension. The parameters start and end 
return the values in which row the loading datas are.
"""
print('-----loading crab datas')
crab_it = read_h5py_chunked(
	'data/crab_precuts.hdf5',
	key='events',
	columns=feature,
	chunksize=250000)
print('start next')
crab, start, end = next(crab_it)

"""
Visualisation of the theta-cuts, too make sure that there are know gammas in 
the signal any more. 
"""
print('-----make hist')
thetaCut = 0.0
plt.title('Theta-cut ' + str(thetaCut) + ' for a better precision')
plt.hist((gammas.theta_deg)**2, np.linspace(0,1.5,50), alpha=0.3, normed=True, label='gammas')
plt.hist((proton.theta_deg)**2, np.linspace(0,1.5,50),alpha=0.3, normed=True, label='simulierte proton')
plt.hist((crab.theta_deg)**2, np.linspace(0,1.5,50), alpha=0.3, normed=True, label='crab')
plt.axvline(x=thetaCut, color='r', linestyle='dashed', linewidth=2)
plt.xlabel(r'$(\Theta)^2$')
plt.ylabel('Hits')
plt.legend(loc='best')
plt.yscale('log')
plt.savefig('plots/theta-cut.pdf')

print('-----make label dataset')
crab = crab[(crab.theta_deg)**2>thetaCut].drop('theta_deg', axis=1)
proton = proton[feature].drop('theta_deg', axis=1)[:180000]
gammas = gammas[feature].drop('theta_deg', axis=1)[:180000]
crab['label'] = 0
proton['label'] = 0
gammas['label'] = 1

print('-----concat gammas and hadrons')
dataSimu = pd.concat([gammas[:180000], proton[:180000]])
dataCrab = pd.concat([gammas[:180000], crab[:180000]])

print('-----pickle data')
dataSimu.to_pickle('data/dataSimu')
dataCrab.to_pickle('data/dataCrab')
