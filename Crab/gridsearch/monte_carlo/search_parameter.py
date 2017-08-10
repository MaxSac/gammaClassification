import matplotlib as mpl
mpl.use('Agg')
exec(open('/home/msackel/Desktop/gammaClassification/programm/tree_optimizer/tree_optimizer.py').read())
exec(open('/home/msackel/Desktop/gammaClassification/programm/theta_cut/theta_cut.py').read())

data = pd.read_pickle('/home/msackel/Desktop/gammaClassification/data/dataSimu')

data.to_pickle('train_set')
optimize_forest('./config.yaml', True)
