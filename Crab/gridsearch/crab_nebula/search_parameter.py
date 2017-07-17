exec(open('/home/msackel/Desktop/gammaClassification/programm/tree_optimizer/tree_optimizer.py').read())
exec(open('/home/msackel/Desktop/gammaClassification/programm/theta_cut/theta_cut.py').read())

data = theta_cut('/home/msackel/Desktop/gammaClassification/data/raw_data/gamma_precuts.hdf5', '/home/msackel/Desktop/gammaClassification/data/raw_data/crab_precuts.hdf5', 0.5)

data.to_pickle('train_set')
optimize_forest('./config.yaml', True)
