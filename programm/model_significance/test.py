from fact.io import read_h5py
from sklearn.ensemble import RandomForestClassifier
import yaml

exec(open('model_significance.py').read())

with open('/home/msackel/Desktop/gammaClassification/config/feature.yaml') as f:
		feature = yaml.load(f)

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
print('max significance: ', model_significance(Tree, hadron_data))
plot_significance(Tree, hadron_data)
plot_on_off_ratio(Tree, hadron_data, 0.8)
print('everything completed *.*')
