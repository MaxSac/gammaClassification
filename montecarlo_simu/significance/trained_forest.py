from sklearn.ensemble import RandomForestClassifier 
from fact.io import read_h5py

exec(open('/home/msackel/Desktop/gammaClassification/programm/theta_cut/theta_cut.py').read())
exec(open('/home/msackel/Desktop/gammaClassification/programm/model_significance/model_significance.py').read())

Tree = RandomForestClassifier(max_depth=15, max_features=7, criterion='entropy', n_estimators=100, n_jobs=10)

with open('/home/msackel/Desktop/gammaClassification/config/feature.yaml') as f:
		feature = yaml.load(f)

columns=list(feature) + ['theta_deg', 'theta_deg_off_1', 'theta_deg_off_2',
	'theta_deg_off_3','theta_deg_off_4', 'theta_deg_off_5']
				
eval_data = pd.read_hdf('/home/msackel/Desktop/gammaClassification/data/raw_data/proton_precuts.hdf5')

print('---Theta**2 = 0.5')
train_data = theta_cut('/home/msackel/Desktop/gammaClassification/data/raw_data/gamma_precuts.hdf5', 
				'/home/msackel/Desktop/gammaClassification/data/raw_data/mrk501_2014_precuts.hdf5', 0.01) 

Tree.fit(train_data.drop('label', axis=1), train_data.label)

plot_significance(Tree, eval_data, path='plots/forest_significance_mc.pdf')
