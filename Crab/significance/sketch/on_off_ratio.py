from fact.io import read_h5py
from fact.analysis import li_ma_significance, split_on_off_source_independent
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

crab_data = read_h5py(
				'/home/msackel/Desktop/gammaClassification/data/raw_data/crab_precuts.hdf5',
				key='events',
				columns=['theta_deg',
						'theta_deg_off_1',
						'theta_deg_off_2',
						'theta_deg_off_3',
						'theta_deg_off_4',
						'theta_deg_off_5',
						]
				)

theta_on = crab_data['theta_deg']
theta_off = pd.concat([crab_data['theta_deg_off_' + str(i)] for i in range(1, 6)])

# plt.style.use('msackel')

plt.figure(figsize=(4.5,3.375))
plt.hist(theta_on**2, range=[0, 0.2], bins=50, histtype='step', label='On')
plt.hist(theta_off**2, range=[0, 0.2], bins=50, alpha=0.6, label='Off', weights=np.full(len(theta_off),  0.2))
plt.plot([0.03,0.03], [0,7000], '-.', label=r'$\theta^2$-Schnitt')
# plt.text(0.008, 4870, r'FP', fontsize=15)
# plt.text(0.008, 5250, r'TP', fontsize=15)
plt.ylim(4700, 6500)
plt.xlabel(r'$\theta^2$/ deg$^2$')
plt.ylabel(r'\SI{100}{\meter}')
plt.legend()
plt.savefig('on_off_ratio.pdf')
