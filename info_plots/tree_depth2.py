from sklearn import tree
import graphviz 
exec(open('/home/msackel/Desktop/gammaClassification/programm/theta_cut/theta_cut.py').read())
exec(open('/home/msackel/Desktop/gammaClassification/programm/model_significance/model_significance.py').read())

with open('/home/msackel/Desktop/gammaClassification/config/feature.yaml') as f:
		feature = yaml.load(f)

train_data = theta_cut('/home/msackel/Desktop/gammaClassification/data/raw_data/gamma_precuts.hdf5', 
	'/home/msackel/Desktop/gammaClassification/data/raw_data/crab_precuts.hdf5', 0.5, length=100000) 

clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(train_data.drop('label',axis=1), train_data.label)

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=train_data.drop('label',axis=1).keys(), class_names=['Hadron','Gamma']) 
graph = graphviz.Source(dot_data) 
graph.render("plots/tree_dept2") 
