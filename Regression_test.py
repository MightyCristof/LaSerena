import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.gaussian_process import GaussianProcess
from astroML.linear_model import NadarayaWatson
from sklearn import cross_validation


print 'Loading DataFrame...'
full_df_name = '/Users/jorgetil/Astro/LaSerena/Project-LSSDS/data/gz2_dr12_full_col.csv'
#train_df_name = '/Users/jorgetil/Astro/LaSerena/Project-LSSDS/data/gz2_dr12_train_samp.csv'
#test_df_name = '/Users/jorgetil/Astro/LaSerena/Project-LSSDS/data/gz2_dr12_test_case.csv'
df_full = pd.read_csv(full_df_name)[:10000]
#df_train = pd.read_csv(train_df_name)
#df_test = pd.read_csv(test_df_name)

print '#### Shapes of tables'
print df_full.shape
#print df_train.shape
#print df_test.shape

print '#### Selecting features from SDSS'
sdss_feat = np.asarray(df_full.keys())[84:85]
print '#### Number of features: %i' % (len(sdss_feat))
print '___________________________________________'


print '#### Splitting data'
y_tar = np.asarray(df_full['t01_smooth_or_features_a01_smooth_weighted_fraction'])
x_vec = np.asarray(df_full.ix[:,84:85])
'''
x_vec_train, x_vec_test, y_tar_train, y_tar_test = cross_validation.train_test_split (x_vec, y_tar, test_size=1./3.)
print "training set = ", x_vec_train.shape, y_tar_train.shape
print "test size = ", x_vec_test.shape, y_tar_test.shape

print '#### Applying unique'
for i in range(len(sdss_feat)):
    feat = str(sdss_feat[i])
    print 'Columnd %i, %s' % (i, feat)
    print 'Starting with %i elements' % (x_vec_train.shape[0])
    aux, idx = np.unique(x_vec_train[:,i], return_index = True)
    #print len(idx)
    #print min(idx)
    x_vec_train = x_vec_train[idx]
    y_tar_train = y_tar_train[idx]
    print 'Ending with %i elements' % (x_vec_train.shape[0])
    print '___________________________________________'

print '#### Final shapes of tables'
print x_vec_train.shape, y_tar_train.shape
'''

print '#### Runing Kernel Regressor'
hs = np.arange(0.001, 0.1, 0.005)
#mse_test = []
#mse_train = []
mean_train, mean_test, std_train, std_test = [], [], [], []
for h in hs:
    print h
    NW_model = NadarayaWatson("gaussian", h = h)
    print 'Fitting'
    #NW_model.fit(x_vec_train, y_tar_train)
    scores_train, scores_test = [], []
    print 'Predicting and doing crossvalidation'
    #y_pre_train = NW_model.predict(x_vec_train[1000:2000])
    #mse_train.append(((y_tar_train[1000:2000] - y_pre_train)**2).sum()/len(y_pre_train))
    ss_train = cross_validation.ShuffleSplit(len(y_tar), n_iter=5, test_size=1./3.)
    for train_inx, test_idx in ss_train:
        NW_model.fit(x_vec[train_inx], y_tar[train_inx])

        y_pre_train = NW_model.predict(x_vec[train_inx])
        scores_train.append(((y_tar[train_inx] - y_pre_train)**2).sum()/len(y_pre_train))

        y_pre_test = NW_model.predict(x_vec[test_idx])
        scores_test.append(((y_tar[test_idx] - y_pre_test)**2).sum()/len(y_pre_test))

    mean_train.append(np.mean(scores_train))
    std_train.append(np.std(scores_train))

    mean_test.append(np.mean(scores_test))
    std_test.append(np.std(scores_test))

print 'Train\n',mean_train
print 'Test\n',mean_test

plt.clf()
plt.errorbar(hs, mean_train, yerr=std_train, label='train', color = 'r')
plt.errorbar(hs, mean_test, yerr=std_test, label='test', color = 'b')
plt.xlabel ("h")
plt.ylabel ("RMSE")
plt.legend(loc = "best")
plt.title ("Gaussian Kernel Regression")
plt.savefig('./Kernerl_regressor_cross.png', dpi = 300)
