from sklearn.ensemble import RandomForestClassifier
from fact.io import read_h5py
exec(open('/home/msackel/Desktop/gammaClassification/programm/theta_cut/theta_cut.py').read())
exec(open('/home/msackel/Desktop/gammaClassification/programm/model_significance/model_significance.py').read())
plt.style.use('msackel')

Tree = RandomForestClassifier(max_depth=15, max_features=7, criterion='entropy', n_estimators=500, n_jobs=30)

with open('/home/msackel/Desktop/gammaClassification/config/feature.yaml') as f:
		feature = yaml.load(f)

eval_data = read_h5py(
				'/home/msackel/Desktop/gammaClassification/data/raw_data/crab_precuts.hdf5',
				key='events',
				columns=list(feature) + [
						'theta_deg',
						'theta_deg_off_1',
						'theta_deg_off_2',
						'theta_deg_off_3',
						'theta_deg_off_4',
						'theta_deg_off_5',
						]
				)

print('---Theta**2 = 0.5')
train_data = pd.read_pickle('/home/msackel/Desktop/gammaClassification/data/dataSimu')

Tree.fit(train_data.drop('label', axis=1), train_data.label)

plot_significance(Tree, eval_data, path='plots/significance_monte.pdf')
