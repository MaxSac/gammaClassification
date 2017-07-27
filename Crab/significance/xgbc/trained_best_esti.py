from xgboost.sklearn import XGBClassifier
from fact.io import read_h5py

exec(open('/home/msackel/Desktop/gammaClassification/programm/theta_cut/theta_cut.py').read())
exec(open('/home/msackel/Desktop/gammaClassification/programm/model_significance/model_significance.py').read())


xgbc = XGBClassifier(			
				max_depth= 3,
				booster= 'gbtree',
				n_jobs= 25,
				eval_metric= 'auc',
				gamma= 0.1,
				min_child_weight= 0.01,
				subsample=0.2,
				learning_rate= 0.2
				)

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

print('---Theta**2 = 0.1')
train_data = theta_cut('/home/msackel/Desktop/gammaClassification/data/raw_data/gamma_precuts.hdf5', 
				'/home/msackel/Desktop/gammaClassification/data/raw_data/mrk501_2014_precuts.hdf5', 0.1) 

xgbc.fit(train_data.drop('label', axis=1), train_data.label)

plot_significance(xgbc, eval_data, path='plots/significance_mrk_bestEsti.pdf')
