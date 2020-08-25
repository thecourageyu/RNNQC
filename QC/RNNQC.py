#!/usr/bin/env python
# coding: utf-8

# In[4]:


from collections import deque, Counter
from datetime import datetime, timedelta
from fbprophet import Prophet
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler

import argparse
import calendar
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys

plt.style.use('ggplot')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1" # first gpu"
from tensorflow.keras.models import load_model
from sklearn import preprocessing

# import tensorflow as tf
# from tensorflow.keras import Input
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.layers import Bidirectional, Dense, GRU, LSTM
# from tensorflow.keras import initializers, regularizers, constraints
# from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, TensorBoard



from lib.dgenerator import dgenerator
from lib.mbuilder import NNBuilder, YZKError, WeightedBinaryCrossntropy
from lib.mdfloader import Dataset
from lib.parallel import runFunctionsInParallel
from lib.textloader import textloader
from lib.perfeval import perfeval as pf

floader   = Dataset.floader
getGI     = Dataset.getGI
qcfparser = Dataset.qcfparser


# In[ ]:


# jupyter nbconvert --ExecutePreprocessor.timeout=90000 --to notebook --execute RNNQC.ipynb
# jupyter nbconvert --to script RNNQC.ipynb


# # Functions

# In[ ]:


df = pd.DataFrame({'Col1': [10, 20, 15, 30, 45],
                   'Col2': [13, 23, 18, 33, 48],
                   'Col3': [17, 27, 22, 37, 52]})
print(df)

# cols = [df.shift(2, axis='index'), df.shift(1, axis='index'), df, df.shift(-1, axis='index')]
# names = ['v1(t-2)', 'v2(t-2)', 'v3(t-2)',
#          'v1(t-1)', 'v2(t-1)', 'v3(t-1)',
#          'v1(t)', 'v2(t)', 'v3(t)',
#          'v1(t+1)', 'v2(t+1)', 'v3(t+1)']
# pda = pd.concat(cols, axis=1)
# pda.columns = names
# print(pda)

pda = dgenerator.series_to_supervised(df, n_in=2, n_out=1, dropnan=False, vnames=['v1', 'v2', 'v3'])
print(pda)


# # Config (for realtime QC)

# In[ ]:


# resd    = "/NAS-DS1515P/users1/T1/res"
# ind     = "/NAS-DS1515P/users1/realtimeQC/ftpdata"
# cmpltd  = ind
# parsed  = "/NAS-DS1515P/users1/realtimeQC/parsed"
# mloutd  = "/NAS-DS1515P/users1/realtimeQC/ml"
# rtqc    = False
# # tperiod = [201912010000, 202005202350]
# tperiod = [202005100000, 202005202350]
# cwbfmt  = ".QPESUMS_STATION.10M.mdf"
# autofmt = ".QPESUMS_STATION.15M.mdf"
# pfmt  = ".QPESUMS_STATION.txt"

# ds = [parsed, mloutd]
# ds.append(parsed)
# ds.append("{0}/LSTM_1".format(mloutd))

# for tmpd in ds:
#     if not os.path.exists(tmpd):
#         os.makedirs(tmpd)


# In[ ]:


# raw_gi, raw_id, nstn = getGI("{0}/stations.txt".format(resd))


# In[ ]:


# qcdtimes = mdfinspector(ind, cmpltd, rtqc, tperiod, cwbfmt, autofmt)


# In[ ]:


# nsize = len(qcdtimes)
# darray = np.ndarray(shape=(nsize, nstn, 4))  # Temp, RH, Pres, Precp
# darray.fill(-999)


# In[ ]:


# start_date = datetime.strptime(str(qcdtimes[0]), "%Y%m%d%H%M")
# end_date   = datetime.strptime(str(qcdtimes[-1]), "%Y%m%d%H%M")
# print(start_date, end_date)


# # Config (for historical data)

# In[ ]:


# ind = "/NAS-129/users1/T1/DATA/YY/ORG/HR1"
# logd = "/home/yuzhe/DataScience/QC/log"
# npyd = "/home/yuzhe/DataScience/dataset"

# if not os.path.exists(logd):
#     os.makedirs(logd)

# if not os.path.exists(npyd):
#     os.makedirs(npyd)
    
# tperiod_train = [2012010101, 2015123124]
# tperiod_test  = [2016010101, 2016123124]

# # stninfo = "/home/yuzhe/CODE/ProgramT1/GRDTools/SRC/RES/GI/RR_analysis_grid_stationlist.txt"
# stninfo = "/home/yuzhe/CODE/ProgramT1/GRDTools/SRC/RES/GI/1500_decode_stationlist_without_space.txt"

# # raw_id = textloader.get_id(stninfo)

# hrdg = dgenerator(ind=ind, gif=stninfo, npyd=npyd)
# hrfs_train = hrdg.hrfgenerator(tperiod_train, n_in=6, n_out=1, mode="train", rescale=True, reformat=True, vstack=True, fnpy=True, generator=False)
# hrfs_test  = hrdg.hrfgenerator(tperiod_test, n_in=6, n_out=1, mode="test", rescale=True, reformat=True, vstack=True, fnpy=True, generator=False)


# In[ ]:


# print(hrfs_train[0].shape, hrfs_train[3].shape)
# print(hrfs_test[0].shape, hrfs_test[3].shape)


# # LSTM with multiple lag timesteps
# # LSTM inputs: A 3D tensor with shape [batch, timestep, feature].

# In[ ]:


4 * ((64 * 128 + 128) + (128 * 128))


# In[ ]:


# target = "Precp"
# mloutd = "/NAS-DS1515P/users1/realtimeQC/ml"
# lstm1d = "{}/LSTM1{}ShortPeriod".format(mloutd, target)
# # lstm1d = "{}/LSTM1TT".format(mloutd)

# if not os.path.exists(lstm1d):
#     os.makedirs(lstm1d)

# fid = open("{0}/lstm1{1}.log".format(lstm1d, target), "w")

# vnames = ["Temp", "RH", "Pres", "Precp"]
# minvals = [-20, 0, 600, 0]
# maxvals = [50, 100, 1100, 220]
# nfeature = len(vnames)
# # tlag * 10 minutes or tlag * hours
# tlag = 6 
# epochs = 50
# batch_size = 50

# trainsize = int(nsize * 0.75)
# trainsize = nsize - keep2test

# fid.write("# of features (variables): {}\n".format(nfeature))
# fid.write("# of timesteps (10 minutes or hour): {}\n".format(tlag))
# fid.write("# of epochs: {}\n".format(epochs))
# fid.write("batch size: {}\n".format(batch_size))
# fid.write("size of training set: {0}\nsample size: {1}\n".format(trainsize, nsize))

# # range check for raw data
# for vidx, vname in enumerate(vnames):
#     print("drop missing values of {}".format(vname))
#     darray[:, :, vidx][np.where(darray[:, :, vidx] < minvals[vidx])] = np.nan


# In[ ]:


# from collections import Counter
# from sklearn.datasets import make_classification
# X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
#                             n_redundant=0, n_repeated=0, n_classes=3,
#                             n_clusters_per_class=1,
#                             weights=[0.01, 0.05, 0.94],
#                             class_sep=0.8, random_state=0)

# print(X.shape, y.shape)
# print(sorted(Counter(y).items()))

# from imblearn.under_sampling import ClusterCentroids
# cc = ClusterCentroids(random_state=0)
# X_resampled, y_resampled = cc.fit_resample(X, y)
# print(sorted(Counter(y_resampled).items()))


# In[ ]:



# from imblearn.under_sampling import RandomUnderSampler
# rus = RandomUnderSampler(random_state=0)
# X_resampled, y_resampled = rus.fit_resample(X, y)
# print(sorted(Counter(y_resampled).items()))


# In[ ]:


# 雨量假設已經被Ｑ出來後
# 再用前後10


# # Train

# In[ ]:


def train(X_train, y_train, epochs, batch_size, mconf, loss="mae", modeld="model", ckptd="ckpt", name="NN", earlystopper=True, lossw=None):
    '''
        X: [nsize, nstn, timestep, feature]
        y: [nsize, nstn, features]
        mconf: {name, units, inshape, outshape, outactfn, batchNormalization, dropouts, activations}
    '''

    if isinstance(loss, str):
        loss = [loss]
    
    timesteps = X_train.shape[1]
    nfeatures = X_train.shape[2]
    
#     timesteps = 6
#     nfeatures = 4

    NN = NNBuilder(modeld=modeld, ckptd=ckptd, name=name)
    if mconf["name"] == "DNNLSTM":
        LSTM, callbacks_, optimizer_ = NN.DNNLSTM(mconf["units"], inshape=(timesteps, nfeatures), outshape=mconf["outshape"], outactfn=mconf["outactfn"], dropouts=mconf["dropouts"], activations=mconf["activations"])
    elif mconf["name"] == "stackedLSTM":
        LSTM, callbacks_, optimizer_ = NN.stackedLSTM(shape=(timesteps, nfeatures), cells=mconf["units"])
    else:
        logging.warning("model name undefined.")
        
    LSTM.summary()
        
    if len(mconf["outactfn"]) == len(mconf["outshape"]) == 2:
        if lossw is None:
            lossw = [1, 1]
        LSTM.compile(loss={"regression_output": loss[0], "classification_output": loss[1]},
                     loss_weights={"regression_output": lossw[0], "classification_output": lossw[1]},
                     metrics={"regression_output": "mae", "classification_output": "accuracy"},
                     optimizer=optimizer_)
    else:
        LSTM.compile(loss=loss[0], optimizer=optimizer_)

    
#     callbacks_ = checkpointer, earlystopper, reduceLR, tb, csvlogger
    
    if not earlystopper:
        callbacks_.pop(1)

    history = LSTM.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks_, validation_split=0.1, verbose=2, shuffle=True)
#     history = LSTM.fit(x=dg, epochs=epochs, batch_size=batch_size, callbacks=callbacks_, verbose=2, shuffle=True)

    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='train')
    ax.plot(history.history['val_loss'], label='test')
    ax.legend(fontsize=14)
    plt.savefig("{}/{}_trainingHistory.png".format(ckptd, name))
    plt.close()
    
    
    return history


# # Fine Tune

# In[6]:


def finetune(X_train, y_train, epochs, batch_size, saved_model, modeld="model", ckptd="ckpt"):
    '''
        X: [nsize, nstn, timestep, feature]
        y: [nsize, nstn, features]
    '''
    
    timesteps = X_train.shape[1]
    nfeatures = X_train.shape[2]
    
    model = NNBuilder.mloader(saved_model)
#     model = load_model(saved_model)
    model.summary()
    
    callbacks_ = NNBuilder()._callbacks(modeld, ckptd, mmonitor="val_loss", emonitor="loss", lmonitor="val_loss", name="ckpt")
    optimizer_ = NNBuilder()._optimizer()
#     stackedLSTM, callbacks_, optimizer_ = NNBuilder(modeld="model", ckptd=ckptd).stackedLSTM(shape=(timesteps, nfeatures), cells=60)
    model.compile(loss="mae", optimizer=optimizer_)
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks_, validation_split=0.1, verbose=2, shuffle=True)
#     history = stackedLSTM.fit(x=dg, epochs=epochs, batch_size=batch_size, callbacks=callbacks_, verbose=2, shuffle=True)

    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='train')
    ax.plot(history.history['val_loss'], label='test')
    ax.legend(fontsize=14)
    plt.savefig("{}/tuningHistory.png".format(ckptd))
    plt.close()
    
    return history


# In[7]:


# hrfs_train = hrdg.hrfgenerator(tperiod_train, n_in=6, n_out=1, mode="train", rescale=True, reformat=True, vstack=True, fnpy=True, generator=False)


# # Data Balance

# In[8]:


# saved_model = "/home/yuzhe/DataScience/QC/model/lstm1_0154_0.009_0.008_202008071814.hdf5"
# epochs = 1500
# batchsize = 100
# finetune(X_finetune, y_finetune, epochs, batchsize, saved_model, modeld="model", ckptd="ckpt")


# # Test

# In[9]:


def dtimes2mnseidx(datetimes, hsystem="01-24", tscale="H"):
    '''
        get start and end idx (seidx) for each YYYYmm
        hsystem = "01-24"  # 01-24 or 00-23
        tscale = "H"  #  D, H or M
    '''

    tformat = "%Y%m%d"

    if tscale == "H":
        sdtime = datetime.strptime(str(math.floor(datetimes[0] / 100.)), tformat)
        edtime = datetime.strptime(str(math.floor(datetimes[-1] / 100.)), tformat)
    elif tscale == "M":
        sdtime = datetime.strptime(str(math.floor(datetimes[0] / 10000.)), tformat)
        edtime = datetime.strptime(str(math.floor(datetimes[-1] / 10000.)), tformat)

    nsize = len(datetimes)
    tdf = pd.DataFrame(datetimes)
    idxdf = pd.DataFrame([i for i in range(nsize)])

    df = pd.concat([tdf, idxdf], axis=1)
    df.columns = ["datetime", "idx"]
    df.set_index("datetime", inplace=True)
    # df.set_index(pd.to_datetime(df["datetime"], format=tformat), inplace=True)


    sdttuple = sdtime.timetuple()
    s_YYYY = sdttuple[0]
    s_mm = sdttuple[1]
    edttuple = sdtime.timetuple()
    e_YYYY = sdttuple[0]
    e_mm = sdttuple[1]


    n_year = e_YYYY - s_YYYY + 1

    seidx = np.zeros((n_year, 12, 2), dtype=np.int32)  # initialize

    mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    mms = [i + 1 for i in range(12)]  # mm
    if hsystem == "01-24":  # HH
        HHs = [i + 1 for i in range(24)]
    else:
        HHs = [i for i in range(24)]
    MMs = [i * 10 for i in range(6)]  # MM


    for yidx in range(n_year):
        YYYY = s_YYYY + yidx

        for mm in mms:
            idx1 = None
            idx2 = None

            mdays_ = mdays[mm - 1]
            if calendar.isleap(YYYY) and mm == 2:
                mdays_ += 1

            # dd
            dds = [i + 1 for i in range(mdays_)]



            # for idx1
            for dd in dds:
                if idx1 is None:
                    if tscale == "H" or tscale == "M":
                        for HH in HHs:
                            if tscale == "M":
                                for MM in MMs:
                                    YmdHM = YYYY * 10**8 + mm * 10**6 + dd * 10**4 + HH * 10**2 + MM
                            else:
                                YmdH = YYYY * 10**6 + mm * 10**4 + dd * 10**2 + HH

                                if YmdH in df.index:
                                    idx1 = df.loc[YmdH]["idx"]
                                    break
                    else:
                        Ymd = YYYY * 10**4 + mm * 10**2 + dd

            # for idx2
            dds.reverse()
            HHs.reverse()
            for dd in dds:
                if idx2 is None:
                    if tscale == "H" or tscale == "M":
                        for HH in HHs:
                            if tscale == "M":
                                for MM in MMs:
                                    YmdHM = YYYY * 10**8 + mm * 10**6 + dd * 10**4 + HH * 10**2 + MM
                            else:
                                YmdH = YYYY * 10**6 + mm * 10**4 + dd * 10**2 + HH

                                if YmdH in df.index:
                                    idx2 = df.loc[YmdH]["idx"]
                                    break       
                    else:
                        Ymd = YYYY * 10**4 + mm * 10**2 + dd
            dds.reverse()
            HHs.reverse()

            seidx[yidx, mm - 1, :] = -999
            if idx1 is not None:
                seidx[yidx, mm - 1, 0] = idx1
                idx1 = None            

            if idx2 is not None:
                seidx[yidx, mm - 1, 1] = idx2
                idx2 = None
                

    idx = []
    for yidx in range(n_year):
        YYYY = s_YYYY + yidx
        for mm in mms:
           idx.append("{0:04d}{1:02d}".format(YYYY, mm)) 
    
    seidx_ = pd.DataFrame(np.reshape(seidx, (-1, 2)))
    seidx_.index = idx
    seidx_.columns = ["sidx", "eidx"]
    
    return [seidx, seidx_]


# In[14]:


def test(X, y_, saved_model, vinfo, scaler, datetimes, stnids, outd, custom_objects=None):
    '''
        X: [nsize, nstn, timestep, feature]
        y: [nsize, nstn, features]
        vinfo: DataFrame
                 Temp      RH    Pres   Precp
            _________________________________
            0     -20       0     600       0
            1      50     100    1100     220
        scaler: inverse transform values from output layer
    '''
    
    print(type(X), type(y_))
    y = y_
    if len(y_) == 2:
        y = y_[0]
    
    nsize = len(datetimes)
    assert nsize == X.shape[0] == y.shape[0]
    nstn = len(stnids)
    assert nstn == X.shape[1] == y.shape[1]
    timestep = X.shape[2]
    nfeature = y.shape[2]
    
#     mname = "LSTM1_woDB_MAE"
#     outd = "/NAS-129/users1/T1/DATA/RNN/QC/{}".format(mname)
       
#     vinfo = pd.DataFrame(hrdg.vrange)
    vnames = vinfo.columns.tolist()
    vrange = vinfo.values
#     scaler = preprocessing.MinMaxScaler()


    if isinstance(scaler, preprocessing.MinMaxScaler):
        scaler.fit(vrange)
    
    
    for product in ["pred", "perfeval"]:
        if product == "perfeval":
            for vname in vnames:
                outd_ = "{}/{}/{}".format(outd, product, vname)
                if not os.path.exists(outd_):
                    os.makedirs(outd_)
        else:
            outd_ = "{}/{}".format(outd, product)
            if not os.path.exists(outd_):
                os.makedirs(outd_)

    if custom_objects is not None:
        model = NNBuilder.mloader(saved_model, custom_objects=custom_objects)
#         model = load_model(saved_model, custom_objects=custom_objects)
    else:
        model = NNBuilder.mloader(saved_model)
#         model = load_model(saved_model)

    model.summary()

    merrors = {"ids":[], "errors": []}
    for idx_, stnid_ in enumerate(stnids):

        y_pred = model.predict(X[:, idx_, :, :])
        if len(y_pred) == 2:
            y_pred = y_pred[0]
        y_pred = scaler.inverse_transform(y_pred)
        y_true = y[:, idx_, :]

        ypdf = pd.DataFrame(y_pred, columns=["{}_pred".format(vname) for vname in vnames])
        ytdf = pd.DataFrame(y_true, columns=["{}_true".format(vname) for vname in vnames])
        outdf = pd.concat([ytdf, ypdf], axis=1)
        outdf.index = datetimes
        outdf.to_csv("{}/pred/{}.csv".format(outd, stnid_))

        outnp = outdf.values
        errnp = np.apply_along_axis(lambda x: x[nfeature:] - x[0:nfeature], 1, outnp)
        errv  = np.nanmean(errnp, axis=0)
        merrors["ids"].append(stnid_)
        merrors["errors"].append(errv)
        
        datetimes_ = np.reshape(np.array(datetimes), (-1, 1))

        y_true = y_true[~np.isnan(y_pred).any(axis=1)]
        datetimes_ = datetimes_[~np.isnan(y_pred).any(axis=1)]
        y_pred = y_pred[~np.isnan(y_pred).any(axis=1)]

        y_pred = y_pred[~np.isnan(y_true).any(axis=1)]
        datetimes_ = datetimes_[~np.isnan(y_true).any(axis=1)]
        y_true = y_true[~np.isnan(y_true).any(axis=1)]

        datetimes_ = datetimes_.ravel().tolist()
        
        if len(datetimes_) <= 0: 
            logging.warning("sample size of {} = {}".format(stnid_, len(datetimes_)))
            continue
        
        seidx = dtimes2mnseidx(datetimes_)
        seidx = seidx[1]
        print(seidx.index.tolist())

        for vidx, vname in enumerate(vnames):
            for Ym in seidx.index.tolist():
                idx1, idx2 = seidx.loc[Ym]
                if idx2 - idx1 <= 10:
                    logging.warning("idx1 = {}, idx2 = {} for {} on {}".format(idx1, idx2, stnid_, Ym))
                    continue

                title = "{}_{}_{}_{}".format(vname, stnid_, datetimes_[idx1], datetimes_[idx2])
                xposi = np.arange(idx2 - idx1 + 1)[0:-1:math.floor((idx2 - idx1 + 1) / 5)].tolist()
                xlabel = [datetimes_[i] for i in xposi]

                kwarg = {"title": title, "xposi": xposi, "xlabel": xlabel}
                print(title)
                pf.tspredict(y_true[idx1:(idx2 + 1), vidx], y_pred[idx1:(idx2 + 1), vidx], outd="{}/perfeval/{}".format(outd, vname), **kwarg)

    
    merrors["ids"].append("total")
    merrors["errors"].append(sum(merrors["errors"]) / len(merrors["errors"]))
    merrors = pd.DataFrame(merrors)
    merrors.to_csv("{}/pred/{}.csv".format(outd, "error_check"))


# In[15]:


a = np.random.random_sample((100, 8))
a[np.random.randint(0, high=100, size=10), 3] = np.nan
# a[~np.isnan(a).any(axis=1]
# print(a)
err = np.apply_along_axis(lambda x: x[4:] - x[0:4], 1, a)
np.nanmean(err, axis=0)


# In[ ]:


a = [1, 3, 4, 5, 6, 52, 23]
sum(a) / len(a)


# In[ ]:


[3.24645079e-01-2.01605185e-01, 1.31765173e-01-8.76716946e-01, 3.18778015e-01 - 2.46106415e-02, 8.81898038e-01 - 2.61233422e-01]


# # LSTM with 60 cells and io with 4 features
# - total parameters = 4 * ((4 * 60 + 60) + (60 * 60)) + (60 * 4 + 4)
# - parameters of input gates (with bias)  = 4 * 60 + 60 + 60 * 60
# - parameters of forget gates (with bias) = 4 * 60 + 60 + 60 * 60
# - parameters of output gates (with bias) = 4 * 60 + 60 + 60 * 60
# - parameters of cell states (with bias)  = 4 * 60 + 60 + 60 * 60
# - parameters of output layer (with bias) = 60 * 4 + 4

# In[43]:


def main(mode, tperiod, gif,
         db=True, n_in=6, n_out=1, dsrc="hrf", 
         mname="NN", mconf=None, epochs=100, batchsize=100, loss="mae", lossw=None,
         ind=None, npyd=None, npysuffix=None, generator=False, 
         saved_model=None, evald=None, custom_objects=None):
    
    dg = dgenerator(ind=ind, gif=gif, npyd=npyd)
    vinfo = pd.DataFrame(dg.vrange)
    
    if mode == "test" and not os.path.exists(evald):
        os.makedirs(evald)
    
    if npyd is not None:
        fnpy = True
    else:
        fnpy = False
        
    if dsrc == "hrf":
        if npysuffix is None:
            npysuffix = mode 
            
        if mode == "test":
            dataset = dg.hrfgenerator(tperiod, n_in=n_in, n_out=n_out, mode=npysuffix, rescale=True, reformat=True, vstack=False, fnpy=fnpy, generator=False)
        else:
            # dim(dataset[0]) = (nsize * nstn, (n_in + n_out) * nfeature)
            dataset = dg.hrfgenerator(tperiod, n_in=n_in, n_out=n_out, mode=npysuffix, rescale=True, reformat=True, vstack=True, fnpy=fnpy, generator=False)
    else:
        pass

    datetimes = dataset[1]
    stnids = dataset[2]
    nsize = len(datetimes)
    nstn = len(stnids)
            
    
    logging.info("# of datetimes = {}".format(nsize))
    logging.info("# of stations = {}".format(nstn))
    
    if mode == "test":
        X = np.reshape(dataset[0][:, :, 0:-4], (-1, nstn, n_in, 4))
        y = dataset[-2]
    else:
        if db:
        
            scaled = np.reshape(dataset[0], (-1, n_in + n_out, 4))
            nsize_ = scaled.shape[0]
            y_class = np.zeros(nsize_)
            logging.debug("shape of scaled: {}".format(scaled.shape))

            precp_ = scaled[:, :, 3]
            scaled0precp = precp_.min()
            
            rains = np.zeros((nsize_), dtype=np.int)
#             rains[np.where(dataset[0][:, -1] > 0)] = 1
            rains[np.where(dataset[0][:, -1] > scaled0precp)] = 1

            rains = np.reshape(rains, (-1, 1))
            
            
#             nminority = precp_[np.any(precp_ != 0, axis=1)].shape[0]
            nminority = precp_[np.any(precp_ != scaled0precp, axis=1)].shape[0]

            nmajority = nsize_ - nminority
#             y_class[np.any(precp_ != 0, axis=1)] = 1
            y_class[np.any(precp_ != scaled0precp, axis=1)] = 1
            logging.debug("rain : others = {} : {} = {} : {}".format(nminority, nmajority, nminority / nminority, nmajority / nminority))

            clscounter = Counter(y_class)
            for key_ in clscounter.keys(): 
                logging.debug("class (scaled): {}, nclass: {}\n".format(key_, clscounter[key_]))

            rus = RandomUnderSampler(random_state=0) 
            print(dataset[0].shape, rains.shape)
            X_resampled, y_resampled = rus.fit_resample(np.hstack([dataset[0], rains]), y_class)  # dim(dataset[0]) = (nsize * nstn, (n_in + n_out) * 4)

            clscounter = Counter(y_resampled)
            for key_ in clscounter.keys(): 
                logging.debug("class (resampled): {}, nclass: {}\n".format(key_, clscounter[key_]))

            if mconf["name"] == "DNNLSTM" and len(mconf["outactfn"]) == len(mconf["outshape"]) == 2:
                X = np.reshape(X_resampled[:, :-5], (-1, n_in, 4))
                y = [X_resampled[:, -5:-1], X_resampled[:, -1]]
            else:
                X = np.reshape(X_resampled[:, :-4], (-1, n_in, 4))
                y = X_resampled[:, -4:]

            logging.debug("shape, X: {}, y: {}, {}".format(X.shape, y[0].shape, y[1].shape))

        else:        
            X = np.reshape(dataset[0][:, :-4], (-1, n_in, 4))
            y = dataset[0][:, -4:]

            logging.debug("shape, X: {}, y: {}".format(X.shape, y.shape))
            
    if mode == "train":
        history = train(X, y, epochs, batchsize, mconf, loss=loss, name=mname)
        return [history, X, y]
    elif mode == "test":
#         scaler = preprocessing.MinMaxScaler()
        scaler = dataset[-1]
        # saved_model = "/home/yuzhe/DataScience/QC/model/lstm1_0154_0.009_0.008_202008071814.hdf5"
#         saved_model = "/home/yuzhe/DataScience/QC/model/lstm1_tune1_0036_0.009_0.011_202008111209.hdf5"

#         mname = "LSTM1_DB_MAE_tune1"
#         outd = "/NAS-129/users1/T1/DATA/RNN/QC/{}".format(mname)

        assert evald is not None
        evald_ = "{}/{}".format(evald, mname)
        test(X, y, saved_model, vinfo, scaler, datetimes, stnids, evald_, custom_objects=custom_objects)
    elif mode == "finetune":
#         saved_model = "/home/yuzhe/DataScience/QC/model/lstm1_0154_0.009_0.008_202008071814.hdf5"
        assert saved_model is not None
        finetune(X, y, epochs, batchsize, saved_model, modeld="model", ckptd="ckpt")    
        
        


# # Model

# In[ ]:


if __name__ == "__main__":
    
    import tensorflow as tf

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    nMB  = 1024 * 8
    
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=nMB)])
        except RuntimeError as e:
            print(e)
    
    tf.random.uniform([1000, 3])
        
    logging.getLogger().setLevel(logging.DEBUG)
    
    ind = "/NAS-129/users1/T1/DATA/YY/ORG/HR1"
    logd = "/home/yuzhe/DataScience/QC/log"
    npyd = "/home/yuzhe/DataScience/dataset"

    if not os.path.exists(logd):
        os.makedirs(logd)

    if not os.path.exists(npyd):
        os.makedirs(npyd)

    tperiod_train = [2012010101, 2015123124]
    tperiod_test  = [2016010101, 2016123124]

    # stninfo = "/home/yuzhe/CODE/ProgramT1/GRDTools/SRC/RES/GI/RR_analysis_grid_stationlist.txt"
    gif = "/home/yuzhe/CODE/ProgramT1/GRDTools/SRC/RES/GI/1500_decode_stationlist_without_space.txt"

    epochs = 70000
    batchsize = 5000
#     dg_ = dgenerator(ind=ind, gif=stninfo, npyd=npyd)
#     dg = dg_.hrfgenerator(tperiod_train, n_in=6, n_out=1, mode="train", rescale=True, reformat=True, vstack=True, fnpy=True, generator=True, batchsize=batchsize)
    
#     train(dg, 7000, batchsize)
    
#     hrdg = dgenerator(ind=ind, gif=gif, npyd=npyd)
#     hrfs_train = hrdg.hrfgenerator(tperiod_train, n_in=6, n_out=1, mode="trainStandard", rescale=True, reformat=True, vstack=True, fnpy=True, generator=False)

#     X_train = hrfs_train[0][:, :-4]
#     X_train = np.reshape(X_train, (-1, 6, 4))  # reshape to [samplesize, timesteps, features]
#     y_train = hrfs_train[0][:, -4:]

#     print(hrfs_train[0].shape, X_train.shape, y_train.shape)
#     train(X_train, y_train, 7000, batchsize)

    # train 
#     loss = YZKError(element_weight=[1 / 6., 1 / 6., 1 / 6., 1 / 2.], penalized=-1)

    regloss = YZKError(penalized=-1)
    regloss = "mae"
    regloss = YZKError()

    clsloss = WeightedBinaryCrossntropy(element_weight=[7, 1])
    loss = [regloss, clsloss]
    lossw = [1, 1]
    mname = "LSTM1_DB_YZKPMAECos"
    mname = "DNNLSTM_DB_MAE_WBC15"    
    mname = "DNNLSTM_DB_YZKMAECos_WBC7_1_drop0p2"
    mconf = {"name": "DNNLSTM", "units": [10, 30, 10, 30], "dropouts": 0.25, "activations": "relu"}
    mconf = {"name": "DNNLSTM", "units": [20, 20, 20, 60], "outactfn": [None, "sigmoid"], "outshape": [4, 1], "dropouts": 0.2, "activations": "relu"}

    ret = main(mode="train", tperiod=tperiod_train, gif=gif, mconf=mconf, npysuffix="trainStandard", db=True, n_in=6, n_out=1, dsrc="hrf", epochs=epochs, batchsize=batchsize, npyd=npyd, mname=mname, loss=loss, lossw=lossw)
    
#     val_loss = 999
#     tolerance= 0.001

#     while (val_loss > tolerance):
#         units = np.random.choice([10, 20, 30, 40, 50], np.random.randint(low=2, high=6, size=1))
#         mname = "DNNLSTM_DB_MAE_{}".format(units)
#         #mconf = {"name": "DNNLSTM", "units": [10, 30, 20, 30], "dropouts": 0.25, "activations": "relu"}
#         mconf = {"name": "DNNLSTM", "units": units, "dropouts": None, "activations": "relu"}

#         history = main(mode="train", tperiod=tperiod_train, gif=gif, mconf=mconf, db=True, n_in=6, n_out=1, dsrc="hrf", epochs=epochs, batchsize=batchsize, npyd=npyd, mname=mname, loss=loss)
#         val_loss = history.history["val_loss"][-1]


    # test
    saved_model = "./model/LSTM1_DB_MAE_2_0055_0.008_0.011_202008111819_2.hdf5"
    mname = "LSTM1_DB_MAE_2"

    saved_model = "./model/LSTM1_DB_YZKLoss_0057_-0.997_-0.991_202008171149.hdf5"
    mname = "LSTM1_DB_YZKLoss"
    
    
    saved_model = "./model/LSTM1_DB_YZKPMAECos_0071_-0.988_-0.980_202008191453.hdf5"
    mname = "LSTM1_DB_YZKPMAECos"
    
    
    saved_model = "DNNLSTM_DB_YZKPMAECos_0022_-0.977_-0.975_202008201418.hdf5"
    mname = "DNNLSTM_DB_YZKPMAECos"
    
    saved_model = "./model/DNNLSTM_DB_YZKMAECos_WBC7_1_drop0p5.hdf5"
    mname = "DNNLSTM_DB_YZKMAECos_WBC7_1_drop0p5"

   
    saved_model = "./model/DNNLSTM_DB_YZKMAECos_WBC7_1_drop0p2.hdf5"
    mname = "DNNLSTM_DB_YZKMAECos_WBC7_1_drop0p2"

    evald = "/NAS-129/users1/T1/DATA/RNN/QC"


    custom_objects = {"YZKError": YZKError, "WeightedBinaryCrossntropy": WeightedBinaryCrossntropy}
#     model = load_model(saved_model)
#     model = load_model(saved_model, custom_objects=custom_objects)
#     y_pred = model.predict(np.random.random_sample((10, 6, 4)))
#     print(y_pred)
#     fnpy = None

#     main(mode="test", ind=ind, tperiod=tperiod_test, gif=gif, npysuffix="testMinMax", db=True, n_in=6, n_out=1, dsrc="hrf", epochs=epochs, batchsize=batchsize, 
    main(mode="test", tperiod=tperiod_test, gif=gif, npysuffix="testStandard", db=True, n_in=6, n_out=1, dsrc="hrf", epochs=epochs, batchsize=batchsize, 
         npyd=npyd, saved_model=saved_model, evald=evald, mname=mname, custom_objects=custom_objects)

    
#     scaled = np.reshape(hrfs_train[0], (-1, 7, 4))
#     nsize = scaled.shape[0]
#     logging.debug("shape of scaled: {}".format(scaled.shape))
#     y_class = np.zeros(nsize)

#     precp_ = scaled[:, :, 3]
#     # np.any(scaled[:, :, 3] != 0, axis=0)

#     nminority = precp_[np.any(precp_ != 0, axis=1)].shape[0]
#     nmajority = nsize - nminority
#     y_class[np.any(precp_ != 0, axis=1)] = 1
#     logging.debug("rain : others = {} : {} = {} : {}".format(nminority, nmajority, nminority / nminority, nmajority / nminority))

#     clscounter = Counter(y_class)
#     for key_ in clscounter.keys(): 
#         logging.debug("class: {}, nclass: {}\n".format(key_, clscounter[key_]))


#     rus = RandomUnderSampler(random_state=0)
#     X_resampled, y_resampled = rus.fit_resample(hrfs_train[0], y_class)

#     clscounter = Counter(y_resampled)
#     for key_ in clscounter.keys(): 
#         logging.debug("class: {}, nclass: {}\n".format(key_, clscounter[key_]))

#     X_train = np.reshape(X_resampled[:, :-4], (-1, 6, 4))
#     y_train = X_resampled[:, -4:]

#     logging.debug("shape, X: {}, y: {}".format(X_train.shape, y_train.shape))
    
#     train(X_train, y_train, epochs, batchsize)


# In[38]:


a = np.reshape(hrfs_train[0], (-1, 7, 4))
a[:, :, 3].min()


# In[20]:


y_true = ret[1]

print(y_true[1].shape)


y_true[1][y_true[1] == 1].shape


# # dgenerator

# In[ ]:


if __name__ == "__main__":
    
    logging.getLogger().setLevel(logging.INFO)
    
    ind = "/NAS-129/users1/T1/DATA/YY/ORG/HR1"
    npyd = "/home/yuzhe/DataScience/dataset"
    
    tperiod_train = [2012010101, 2015123124]
    tperiod_test  = [2016010101, 2016123124]

    # stninfo = "/home/yuzhe/CODE/ProgramT1/GRDTools/SRC/RES/GI/RR_analysis_grid_stationlist.txt"
    gif = "/home/yuzhe/CODE/ProgramT1/GRDTools/SRC/RES/GI/1500_decode_stationlist_without_space.txt"
    
    hrdg = dgenerator(ind=ind, gif=gif, npyd=npyd)
    hrfs_train = hrdg.hrfgenerator(tperiod_train, n_in=6, n_out=1, mode="train", rescale=True, reformat=True, vstack=True, fnpy=True, generator=False)

    
#     X = np.reshape(hrfs_test[0][:, :, 0:-4], (-1, nstn, 6, 4))
#     y = hrfs_test[-1]
#     print("shape, X: {}, y: {}".format(X.shape, y.shape))

#     datetimes = hrfs_test[1]
#     stnids = hrfs_test[2]


# In[ ]:


hrfs_train[0][:, -1]


# In[ ]:


rainsimes = hrfs_train[1]
nsize = len(datetimes)
stnids = hrfs_train[2]
nstn = len(stnids)
print(nsize, nstn)
hrdata = hrfs_train[-1]

rains = np.zeros((nsize * nstn, 1), dtype=np.int)
hrstacked = np.reshape(hrdata, (-1, 4))
# precpd = pd.DataFrame(hrstacked[:, -1], columns=["precp"])

precpd = hrstacked[:, -1]
rains[np.where(precpd > 0)] = 1

clscounter = Counter(rains.ravel())
for key_ in clscounter.keys(): 
    logging.debug("class (scaled): {}, nclass: {}\n".format(key_, clscounter[key_]))


# In[ ]:


norain_ = rains[rains == 0].shape[0]
rain_ = rains[rains == 1].shape[0]
assert norain_ + rain_ == nsize * nstn
print("not rain: {}, rain: {}, total: {}".format(norain_, rain_, nsize * nstn))


# In[ ]:


norain_ / rain_


# In[ ]:


# precpd.describe()
print("shape of precpd: {}".format(precpd.shape))
precpd.dropna(inplace=True)
print("shape of precpd (dropna): {}".format(precpd.shape))

precpd.describe()
precpd.hist(bins=30)


# In[ ]:


hrstacked


# In[ ]:


precpd = precpd.values


# In[ ]:


prepgt0 = pd.DataFrame(precpd[np.where(precpd > 0)])
prepgt0.describe()

fig, ax = plt.subplots(figsize=(16, 10))
# ax.set_xlim([-1, 100])
n, bins, patches = ax.hist(prepgt0.values, bins=10)
print(n, bins)


# ## LSTM inputs: A 3D tensor with shape [batch, timesteps, feature].

# ## 2. LSTM Model

# In[ ]:


from tensorflow.keras.utils import plot_model
model, callbacks_, optimizer_ = NNBuilder().DNNLSTM([10, 20, 20, 10], inshape=[6, 4], outshape=4)
model.summary()
plot_model(model, to_file="model.png", show_shapes=True)
# plot_model(
#     model, to_file='model.png', show_shapes=False, show_layer_names=True,
#     rankdir='TB', expand_nested=False, dpi=96
# )


# In[ ]:


epochs = 50
batch_size = 50


# In[ ]:


# import keras
import pydot as pyd
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# keras.utils.vis_utils.pydot = pyd

#Visualize Model

def visualize_model(model):
    return SVG(model_to_dot(model, show_layer_names=True, show_shapes=True, dpi=60).create(prog='dot', format='svg'))
#create your model
#then call the function on your model
visualize_model(lstm)


# In[ ]:


# from keras.utils import plot_model

import pydotplus
from keras.utils.vis_utils import model_to_dot, pydot
# keras.utils.vis_utils.pydot = pydot
plot_model(lstm, to_file='model.png')
# plot_model(lstm)


# In[ ]:


a = stats.anderson(residual_, dist='norm')
print(a)

stests, stestp = stats.shapiro(residual_)
print(stestp)

# the null hypothesis that a sample comes from a normal distribution.
ntests, ntestp = stats.normaltest(residual_)
print(ntestp)

n, bins = np.histogram(residual_, bins=30)
print(residual_)
print(sum(n / len(residual_)))
print(n / len(residual_))

n, bins = np.histogram(residual_, bins=30)

widths = bins[1:] - bins[:-1]
print(widths)

aa = (n * widths)
print(aa)

# plt.bar(bins[1:], n / sum(n))



plt.bar(bins[1:], n / len(residual_))
plt.show()
plt.hist(residual_, bins=30)


# In[ ]:


residual_ = inv_yhat - inv_y
mu_ = np.mean(residual_)
std_ = np.std(residual_)

# the null hypothesis that the data was drawn from a normal distribution.
stests, stestp = stats.shapiro(residual_)

# the null hypothesis that a sample comes from a normal distribution.
ntests, ntestp = stats.normaltest(residual_)

props = dict(boxstyle='round', facecolor='red', alpha=0.5)

# descstat = "(25%q, 50%q, 75%q) = ({0:>6.2f}, {1:>6.2f}, {2:>6.2f})\n($\mu$, $\sigma$) = ({3:7.3f}, {4:7.3f})\n".format(np.quantile(residual_, 0.25), np.quantile(residual_, 0.75), np.quantile(residual_, 0.75), np.mean(residual_), np.std(residual_))
descstat = "($\mu$, $\sigma$) = ({0:7.3f}, {1:7.3f})\n(shapiro, ntest) = ({2:7.3f}, {3:7.3f})".format(mu_, std_, stestp, ntestp)
print(descstat)

minv = min(np.min(inv_y), np.min(inv_yhat))
maxv = max(np.max(inv_y), np.max(inv_yhat))

plt.figure(figsize=(16, 10))
ax1 = plt.subplot2grid(shape=(7, 4), loc=(0, 0), rowspan=3, colspan=4)
# fig
ax1.plot(inv_y, label='$y$') 
ax1.plot(inv_yhat, label='$\hat{y}_{LSTM}$')
ax1.legend()

plt.title(raw_id[stnidx])

ax2 = plt.subplot2grid(shape=(7, 4), loc=(3, 0), rowspan=2, colspan=4)
ax2.plot(residual_, label='residual', color='blue')
ax2.plot(np.zeros(residual_.shape[0]), '--', color='red')
ax2.legend()

ax2.text(residual_.shape[0], min(residual_), 'RMSE = {0:>7.3f}'.format(rmse),
         verticalalignment='bottom', horizontalalignment='right',
         color='black', fontsize=10, bbox=props)

# figure 3, error distribution
ax3 = plt.subplot2grid(shape=(7, 4), loc=(5, 0), rowspan=2, colspan=2)
n, bins, patches = ax3.hist(residual_, bins=30, density=True, stacked=True, label='residual', facecolor='g', alpha=0.75)
# n, bins, patches = ax3.hist(residual_, density=True, stacked=True, label='residual', facecolor='g', alpha=0.75)

resx = np.arange(min(residual_), max(residual_), 0.001)
resnpdf = norm(mu_, std_).pdf(resx)
ax3.plot(resx, resnpdf, label='normal', color='red')
ax3.text(bins[int(len(bins) * 0.6)], max(n) * 0.6, s=descstat, bbox=props)

# ax3.plot(np.zeros(residual_.shape[0]), '--', color='red')
ax3.legend()

ax4 = plt.subplot2grid(shape=(7, 4), loc=(5, 2), rowspan=2, colspan=2)
ax4.scatter(inv_y, inv_yhat, label='residual', color='skyblue')
ax4.plot([minv, maxv], [minv, maxv], '--', color='red')

plt.tight_layout()
plt.show()
# print(reframed.index.values[-inv_y.shape[0]:])
# plt.savefig("{}/LSTM_1/LSTM_1_{}.png".format(mloutd, raw_id[stnidx]))
plt.close()

print(np.min(inv_y), np.min(inv_yhat))


# In[ ]:


"{}/LSTM_1/LSTM_1_{}.png".format(mloutd, raw_id[stnidx])


# # Facebook Prophet Model

# In[ ]:


keepsize = 0.75
id_idx = 10
# dt_idx =

qcdtimes_ = [datetime.strptime(str(dtime), "%Y%m%d%H%M") for dtime in qcdtimes]
samplesize = len(qcdtimes_)
trainsize = int(samplesize * keepsize)
testsize = samplesize - trainsize

train_ds = qcdtimes_[0:trainsize]
train_y  = darray[0:trainsize, id_idx, 0]

test_ds  = qcdtimes_[trainsize:]
test_y  = darray[trainsize:, id_idx, 0]

train_df = pd.DataFrame(data={"ds": train_ds, "y": train_y})
test_df  = pd.DataFrame(data={"ds": test_ds, "y": test_y})


# In[ ]:


m = Prophet(daily_seasonality=True) # the Prophet class (model)
m.fit(train_df)


# In[ ]:


future = m.make_future_dataframe(periods=math.ceil(testsize / 6), freq="H") # we need to specify the number of days in future
prediction = m.predict(future)


# In[ ]:


prediction.ds


# In[ ]:


m.plot(prediction)
plt.title("Prediction of the Google Stock Price using the Prophet")
plt.xlabel("Date")
plt.ylabel("Close Stock Price")
plt.show()


# In[ ]:


prediction.columns


# In[ ]:


prediction["ds"]


# In[ ]:


samplesize


# In[ ]:


test_df


# In[ ]:


np.where(prediction["ds"] == test_df["ds"][0])


# In[ ]:


test_df["ds"][0]


# In[ ]:


a = prediction["ds"]


# In[ ]:


pred = pd.merge(prediction, test_df)
pred["resi"] = pred["yhat"] - pred["y"]


# In[ ]:


pred


# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(16, 10))
m.plot(prediction, ax=axes[0, 0])
pred.plot.scatter(x='ds', y='y', ax=axes[0, 0], color='red')
pred.plot.scatter(x='y', y='yhat', ax=axes[0, 1])
axes[0, 1].plot([0, 30], [0, 30])
pred.plot(x='ds', y='resi', ax=axes[1, 1])
pred.plot(x='ds', y='yhat', ax=axes[1, 0])
pred.plot(x='ds', y='y', ax=axes[1, 0])


# In[ ]:


df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'fool'],
                    'value': [1, 2, 3, 3]})
df2 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz'],
                    'value1': [5, 6, 7]})

