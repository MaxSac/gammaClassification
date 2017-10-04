import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import numpy as np
from fact.io import read_h5py
plt.style.use('msackel')

hadron = read_h5py(
				'/home/msackel/Desktop/gammaClassification/data/raw_data/crab_precuts.hdf5',
				key='events',
				columns=['theta_deg']
				)

gamma = pd.read_hdf(
				'/home/msackel/Desktop/gammaClassification/data/raw_data/gamma_precuts.hdf5',
				key='events'
				)

hadron['label'] = 0
gamma['label'] = 1




fig = plt.figure(figsize=(5.4,4.05)) 
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) # Verzerrte Graphen

ax1 = plt.subplot(gs[0])
histHadron = plt.hist(hadron.theta_deg**2, range=[0,3], bins=20, histtype='step', label='Proton')
histGammas = plt.hist(gamma.theta_deg**2, range=[0,3], bins=20, histtype='step', label='Gamma')
#plt.yscale('log')
plt.legend()
plt.setp(ax1.get_xticklabels(), visible=False) # make these tick labels invisible

ax2 = plt.subplot(gs[1], sharex=ax1)
plt.bar(np.linspace(0.0,2.88,20)+0.06, histGammas[0]/histHadron[0], width= 0.12)
plt.yscale('log')
plt.xlabel(r'$\theta^2 /$ deg$^2$')

plt.savefig('theta_cut.pdf')
