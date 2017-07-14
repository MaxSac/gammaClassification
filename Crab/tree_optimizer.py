import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import yaml
plt.style.use('ggplot')

with open('config.yaml') as f:
	CONF = yaml.load(f)

DATA = pd.read_pickle(CONF['path'])

Tree = RandomForestClassifier(**CONF['set_params'])

clf = GridSearchCV(
		Tree, 
		param_grid= CONF['loop_params'], 
		**CONF['grid_params']
		)

clf.fit(DATA.drop('label', axis=1), DATA.label)

best_esti = pd.DataFrame(clf.cv_results_)
best_esti.to_pickle('bestEsti')

FEATURE_1 = best_esti.param_max_depth
FEATURE_2 = best_esti.param_max_features

x = np.unique(FEATURE_1.values)
y = np.unique(FEATURE_2.values)
z = best_esti.mean_test_score.values.reshape(len(y),len(x))

plt.imshow(z)
plt.xticks(range(len(np.unique(FEATURE_1.values))), x)
plt.yticks(range(len(np.unique(FEATURE_2.values))), y)
plt.xlabel(FEATURE_1.name)
plt.ylabel(FEATURE_2.name)
plt.title('Feature grid to search best roc_auc score')
plt.colorbar()
plt.savefig('parameter_grid.pdf')
