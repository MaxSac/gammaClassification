from sklearn.ensemble import RandomForestClassifier
from fact.io import read_h5py
from matplotlib import pyplot as plt
plt.style.use('msackel')
exec(open('/home/msackel/Desktop/gammaClassification/programm/theta_cut/theta_cut.py').read())
exec(open('/home/msackel/Desktop/gammaClassification/programm/model_significance/model_significance.py').read())

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

eval_data['pred']= Tree.predict_proba(eval_data[feature])[:,1]

selected = eval_data.query('pred >= 0.8')
theta_on = selected.theta_deg
theta_off = pd.concat([selected['theta_deg_off_' + str(i)] for i in range(1, 6)])

plt.figure(figsize=(4.5,3.375))
plt.hist(theta_on**2, range=[0, 0.2], bins=50, histtype='step', label='On')
plt.hist(theta_off**2, range=[0, 0.2], bins=50, alpha=0.6, label='Off', weights=np.full(len(theta_off),  0.2))
plt.plot([0.03,0.03], [0,7000], '-.', label=r'$\theta^2$-Schnitt')
plt.text(0.008, 430, r'FP', fontsize=15)
plt.text(0.008, 750, r'TP', fontsize=15)
plt.ylim(300, 1800)
plt.xlabel(r'$\theta^2$/ deg$^2$')
plt.ylabel(r'Hits')
plt.legend()
plt.savefig('on_off_ratio2.pdf')

