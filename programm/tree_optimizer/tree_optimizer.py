import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import yaml
plt.style.use('ggplot')

def optimize_forest(config, visualisation= False):
	'''
	It's a function which search for the best settings for a Forest with given 
	parameters. The parameters are given in a config.yaml file. The reesponse is a 
	plot and a file with the scores of the different settings.
	Paramters:
		config: string
			path to config.yaml file to optimize the trees
		visualisation: boolean
			Make a parmeter grid of two features an give the score as a color back
	Returns:
		best_esti: pd.DataFrame
			DataFrame with scores paramters and ranking of the diferent classifier
	'''
	with open(config) as f:
		CONF = yaml.load(f)
	DATA = pd.read_pickle(CONF['path'])
	
	'''
	Make a gridsearch over given parameters and fit the different models.
	'''
	Tree = RandomForestClassifier(**CONF['set_params'])
	clf = GridSearchCV(
			Tree, 
			param_grid= CONF['loop_params'], 
			**CONF['grid_params']
			)
	clf.fit(DATA.drop('label', axis=1), DATA.label)
	
	'''
	Save the scorrings of the different parameters.
	'''
	best_esti = pd.DataFrame(clf.cv_results_)
	best_esti.to_pickle('bestEsti')
	
	if(visualisation == True):
		FEATURE_1 = best_esti['param_' + CONF['plot_feature']['feature_one']]
		FEATURE_2 = best_esti['param_' + CONF['plot_feature']['feature_two']]
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
	
	return best_esti
