exec(open('theta_cut.py').read())
data = theta_cut('/home/msackel/Desktop/gammaClassification/data/raw_data/gamma_precuts.hdf5', '/home/msackel/Desktop/gammaClassification/data/raw_data/crab_precuts.hdf5', 0.5)
print(data)
