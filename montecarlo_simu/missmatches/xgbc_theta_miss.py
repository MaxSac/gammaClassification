from xgboost.sklearn import XGBClassifier
from fact.io import read_h5py

exec(open('/home/msackel/Desktop/gammaClassification/programm/theta_cut/theta_cut.py').read())
exec(open('/home/msackel/Desktop/gammaClassification/programm/model_significance/model_significance.py').read())

xgbc = XGBClassifier(				
				max_depth= 1,
				learning_rate= 0.5,
				n_estimators= 100,
				booster= 'gbtree', 
				gamma= 0.5,
				n_jobs= 15,
				reg_lambda= 0.1,
				subsample= 0.20,
				eval_metric= 'auc'
				)

with open('/home/msackel/Desktop/gammaClassification/config/feature.yaml') as f:
		feature = yaml.load(f)

columns=list(feature) + ['theta_deg', 'theta_deg_off_1', 'theta_deg_off_2',
	'theta_deg_off_3','theta_deg_off_4', 'theta_deg_off_5']
				
eval_data = pd.read_hdf('/home/msackel/Desktop/gammaClassification/data/raw_data/proton_precuts.hdf5')

print('---Theta**2 = 0.5')
train_data = theta_cut('/home/msackel/Desktop/gammaClassification/data/raw_data/gamma_precuts.hdf5', 
				'/home/msackel/Desktop/gammaClassification/data/raw_data/mrk501_2014_precuts.hdf5', 0.01) 

xgbc.fit(train_data.drop('label', axis=1), train_data.label)

eval_data['prediction'] = xgbc.predict(eval_data[feature])
plt.hist(eval_data[eval_data['prediction']==0]['theta_deg']**2, bins=50, range=[0,9], histtype='step', label='predicted hadron')
plt.hist(eval_data['theta_deg']**2, bins=50, range=[0,9], histtype='step', label='MC hadron')
plt.legend(loc='best')
plt.savefig('plots/xgbc_theta_miss.pdf')
