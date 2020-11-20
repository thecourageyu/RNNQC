#!/opt/anaconda3/bin/python3
# coding: utf-8

# In[ ]:




# In[2]:


import argparse
import calendar
import gc
import logging
import math
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('ggplot')

from collections import deque, Counter
from datetime import datetime, timedelta
from fbprophet import Prophet
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler, TomekLinks

# from tensorflow.keras.models import load_model
from sklearn import preprocessing

import tensorflow as tf
# from tensorflow.keras import Input
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.layers import Bidirectional, Dense, GRU, LSTM
# from tensorflow.keras import initializers, regularizers, constraints
# from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from lib.dgenerator import dgenerator
from lib.mbuilder import NNBuilder, YZKError, WeightedBinaryCrossntropy, ChangeableLossw, TPCNN1D, TPAttention
from lib.mdfloader import Dataset
from lib.parallel import runFunctionsInParallel
from lib.textloader import textloader
from lib.perfeval import perfeval as pf
from lib.perfeval import getClasses

floader   = Dataset.floader
getGI     = Dataset.getGI
qcfparser = Dataset.qcfparser


# In[ ]:


# jupyter nbconvert --ExecutePreprocessor.timeout=90000 --to notebook --execute RNNQC.ipynb
# jupyter nbconvert --to script RNNQC.ipynb


# # Functions

# # LSTM with multiple lag timesteps
# # LSTM inputs: A 3D tensor with shape [batch, timestep, feature].

# # Train

# In[3]:


def ratio_multiplier(y):
    '''
        multiplier: resampling ratio of sample for some classes
    '''
    
    multiplier={0: 0.1, 1: 1.0}
    
    target_stats = Counter(y)
    for key, value in target_stats.items():
        if key in multiplier:
            target_stats[key] = int(value * multiplier[key])
    return target_stats


# In[4]:


# def train(X_train, y_train, epochs, batch_size, mconf, loss="mae", modeld="model", ckptd="ckpt", name="NN", earlystopper=True, lossw=None):

def train(X_train, y_train, epochs, batch_size, mconf, modeld="model", ckptd="ckpt", name="NN"):
    '''
        positional:
            X_train: array with shape [nsize, nstn, timestep, feature]
            y_train: [nsize, nstn, features]
            epochs
            batch_size
            mconf: {name, units, inshape, outshape, outactfn, loss, lossw, metric, batchNormalization, dropouts, activations, earlystopper, dropout, recurrent_dropout, changeable_lossw}
        keyword:
            modeld: directory of saved model 
            ckptd: directory of check point
            name: name for output
    '''

    loss = mconf["loss"]
    if isinstance(loss, str):
        loss = [loss]
    
    timestep = X_train.shape[1]
    nfeature = X_train.shape[2]
    
    NN = NNBuilder(modeld=modeld, ckptd=ckptd, name=name)
    if mconf["name"] == "DNNLSTM":
        model, callbacks_, optimizer_ = NN.DNNLSTM(units=mconf["units"], inshape=(timestep, nfeature), outshape=mconf["outshape"], outactfn=mconf["outactfn"], dropouts=mconf["dropouts"], activations=mconf["activations"])
    elif mconf["name"] == "stackedLSTM":
        model, callbacks_, optimizer_ = NN.stackedLSTM(cells=mconf["units"], inshape=(timestep, nfeature), outshape=mconf["outshape"], outactfn=mconf["outactfn"], dropout=mconf["dropout"], recurrent_dropout=mconf["recurrent_dropout"])
    elif mconf["name"] == "bidirectionalLSTM":
        model, callbacks_, optimizer_ = NN.bidirectionalLSTM(cells=mconf["units"], inshape=(timestep, nfeature), outshape=mconf["outshape"], outactfn=mconf["outactfn"], dropout=mconf["dropout"], recurrent_dropout=mconf["recurrent_dropout"], merge_mode='concat')
    elif mconf["name"] == "CNN1D":
        model, callbacks_, optimizer_ = NN.CNN1D(filters=mconf["units"], inshape=(timestep, nfeature), outshape=mconf["outshape"], outactfn=mconf["outactfn"], activations=mconf["activations"])
    else:
        logging.warning("undefined model name.")
        
    model.summary()
    
    if "metric" not in mconf.keys():
        mconf["metric"] = ["mae", tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None)]
    
    if "outactfn" in mconf.keys() and len(mconf["outactfn"]) == len(mconf["outshape"]) == 2:
        lossw = mconf["lossw"]
        if lossw is None:
            lossw = [1, 1]
        
#         lossw1 = lossw[0]
#         lossw2 = lossw[1]
            
        if "changeable_lossw" in mconf.keys():
            lossw = tf.Variable(lossw)
            wmultiplier = tf.Variable(mconf["changeable_lossw"])
            clw = ChangeableLossw(lossw, wmultiplier)

            
        logging.info("****** loss_weights: {}".format(lossw))
        model.compile(loss={"Loss1": loss[0], "Loss2": loss[1]},
                      loss_weights={"Loss1": lossw[0], "Loss2": lossw[1]},
#                       loss_weights=[lossw[0], lossw[1]],
                      metrics={"Loss1": mconf["metric"][0], "Loss2": mconf["metric"][1]},
                      optimizer=optimizer_)
    else:
        model.compile(loss=loss[0], metrics={"Loss1": mconf["metric"][0]}, optimizer=optimizer_)

    
#     callbacks_ = checkpointer, earlystopper, reduceLR, tb, csvlogger
    
    if not mconf["earlystopper"]:
        callbacks_.pop(1)
        
    if isinstance(y_train, list):
        logging.info("shape of X_train: {}, y_train[0]: {}, y_train[1]: {}".format(X_train.shape, y_train[0].shape, y_train[1].shape))
    else:
        logging.info("shape of X_train: {}, y_train: {}".format(X_train.shape, y_train.shape))

        
    if "changeable_lossw" in mconf.keys():
        logging.info("****** changeable weights activate: {}".format(mconf["changeable_lossw"]))

        print("****** changeable weights activate: {}".format(mconf["changeable_lossw"]))
        print("****** changeable weights activate: {}".format(mconf["changeable_lossw"]))
        print("****** changeable weights activate: {}".format(mconf["changeable_lossw"]))

        
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks_ + [clw], validation_split=0.3, verbose=1, shuffle=True)
    else:
#         history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks_, validation_split=0.3, verbose=1, shuffle=False)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks_, validation_split=0.3, verbose=1, shuffle=True)

#     history = LSTM.fit(x=dg, epochs=epochs, batch_size=batch_size, callbacks=callbacks_, verbose=2, shuffle=True)

    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='train')
    ax.plot(history.history['val_loss'], label='test')
    ax.legend(fontsize=14)
    plt.savefig("{}/{}_trainingHistory.png".format(ckptd, name))
    plt.close()
    
    
    return history


# # Fine Tune

# In[5]:


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


# In[6]:


# hrfs_train = hrdg.hrfgenerator(tperiod_train, n_in=6, n_out=1, mode="train", rescale=True, reformat=True, vstack=True, fnpy=True, generator=False)


# # Test

# In[7]:


def dtimes2mnseidx(datetimes, hsystem="01-24", tscale="H"):
    '''
        get start and end idx (seidx) for each YYYYmm
        hsystem = "01-24"  # 01-24 or 00-23
        tscale = "H"  #  D, H or M
    '''

    datetimes_ = np.copy(datetime)
    
    tformat = "%Y%m%d"

    if tscale == "H":
        sdtime = datetime.strptime(str(math.floor(datetimes[0] / 100.)), tformat)
        edtime = datetime.strptime(str(math.floor(datetimes[-1] / 100.)), tformat)
    elif tscale == "M":
        if np.array([isinstance(datetimes[i], datetime) for i in range(2)]).all():
            datetimes = [int(i.strftime("%Y%m%d%H%M")) for i in datetimes]
#             print("dtimes2mnseidx-18: \n", datetimes)
        else:
            logging.Error("dtimes2mnseidx-16: np.array([isinstance(datetimes[i], datetime) for i in range(2)]).all() == False.")
            sys.exit(-1)
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
    edttuple = edtime.timetuple()
    e_YYYY = edttuple[0]
    e_mm = edttuple[1]

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

            if (YYYY * 100 + mm < s_YYYY * 100 + s_mm) or (e_YYYY * 100 + e_mm < YYYY * 100 + mm):
                seidx[yidx, mm - 1, :] = -999
                continue
            
            # for idx1
            for dd in dds:
                if idx1 is None:
                    if tscale == "H" or tscale == "M":
                        for HH in HHs:
                            if tscale == "M":
                                for MM in MMs:
                                    YmdHM = YYYY * 10**8 + mm * 10**6 + dd * 10**4 + HH * 10**2 + MM
                                    if YmdHM in df.index:
                                        idx1 = df.loc[YmdHM]["idx"]
                                        break
                                if idx1 is not None:
                                    break
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
            MMs.reverse()
            for dd in dds:
                if idx2 is None:
                    if tscale == "H" or tscale == "M":
                        for HH in HHs:
                            if tscale == "M":
                                for MM in MMs:
                                    YmdHM = YYYY * 10**8 + mm * 10**6 + dd * 10**4 + HH * 10**2 + MM

                                    if YmdHM in df.index:
                                        idx2 = df.loc[YmdHM]["idx"]
                                        break   
                                if idx2 is not None:
                                    break
                            else:
                                YmdH = YYYY * 10**6 + mm * 10**4 + dd * 10**2 + HH

                                if YmdH in df.index:
                                    idx2 = df.loc[YmdH]["idx"]
                                    break       
                    else:
                        Ymd = YYYY * 10**4 + mm * 10**2 + dd
            dds.reverse()
            HHs.reverse()
            MMs.reverse()

            seidx[yidx, mm - 1, :] = -999
            if idx1 is not None:
                seidx[yidx, mm - 1, 0] = idx1
                idx1 = None            

            if idx2 is not None:
                seidx[yidx, mm - 1, 1] = idx2
                idx2 = None
                
#             gc.collect()
                
    idx = []
    for yidx in range(n_year):
        YYYY = s_YYYY + yidx
        for mm in mms:
           idx.append("{0:04d}{1:02d}".format(YYYY, mm)) 
    
    seidx_ = pd.DataFrame(np.reshape(seidx, (-1, 2)))
    seidx_.index = idx
    seidx_.columns = ["sidx", "eidx"]
    
    return [seidx, seidx_]


# In[ ]:


def test(X, y_, saved_model, vinfo, scaler, datetimes, stnids, ndel=0, outd=None, custom_objects=None, classify=None, task=None, hsystem="01-24", tscale="H"):
    '''
        - X: [nsize, nstn, timestep, feature]
        - y_: [nsize, nstn, ntarget], [[nsize, nstn, ntarget], [nsize, nstn, nclass]] or [nsize, nstn, nclass] 
        - vinfo: DataFrame (specify predict which variables for y_pred)
                 Temp      RH    Pres   Precp
            _________________________________
            0     -20       0     600       0
            1      50     100    1100     220
        - scaler: inverse transform values from output layer
        - ndel: delete first n data
        - classify: be used to do "class to step" by perfeval
        - tasks
            1. task="classification", then y_class=y_, get prediction y_class_pred, return y_clsout
            2. task="class2step", then y=y_[0] and y_class=y_[1], get prediction y_class_pred, return y_clsout and get class2step in perfeval
            3. task=None, then y=y_, get predictions y_pred, return y_valout
            4. task=None and len(y_)=2, then y=y_[0] and y_class=y_[1], get predictions y_pred and y_class_pred, return [y_valout, y_clsout]
        
        Note.
            - need to search mask support when input is missing or not complete
    '''
    
    logging.debug("shape of X = {}".format(X.shape))
    logging.debug("len(y_) = {}".format(len(y_)))
    
    if task == "classification":  # shape of y_: [nsize, nstn, nclass] 
        y_class = y_
        nclass = y_class.shape[2]
        logging.debug("shape of y_class = {} for classification.".format(y_class.shape))
    elif task == "class2step":  # shape of y_: [[nsize, nstn, ntarget], [nsize, nstn, nclass]]
        y = y_[0]
        y_class = y_[1]
        nclass = y_class.shape[2]
        logging.debug("shape of y_class = {} for class2step.".format(y_class.shape))
    else:
        if isinstance(y_, list) and len(y_) >= 2:  # shape of y_: [[nsize, nstn, ntarget], [nsize, nstn, nclass]]
            y = y_[0]
            y_class = y_[1]
            nclass = y_class.shape[2]
            logging.debug("test-38: shape of y_class = {}.".format(y_class.shape))
        else:  # shape of y_: [nsize, nstn, ntarget] 
            y = y_
        logging.debug("shape of y = {}.".format(y.shape))

    nsize = len(datetimes)
    nstn = len(stnids)

    if task == "classification" or task == "class2step":
        assert nsize == X.shape[0] == y_class.shape[0]
        assert nstn == X.shape[1] == y_class.shape[1]
    else:
        assert nsize == X.shape[0] == y.shape[0]
        assert nstn == X.shape[1] == y.shape[1]
    
    timestep = X.shape[2]
    nfeature = X.shape[3]
    
    vnames = vinfo.columns.tolist()
    vrange = vinfo.values
    ntarget = len(vnames)
    
    # declare array for storing predictions 
    if task == "classification" or task == "class2step":
        assert ntarget == 1
        y_clsout = np.ndarray((nsize, nstn, nclass))
    else:
        y_valout = np.ndarray((nsize, nstn, ntarget))
        if isinstance(y_, list) and len(y_) >= 2:
            y_clsout = np.ndarray((nsize, nstn, nclass))
    
#     if isinstance(scaler, preprocessing.MinMaxScaler):
#         scaler.fit(vrange)
        
    if outd is not None:
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
    else:
        model = NNBuilder.mloader(saved_model)

    model.summary()

    merrors = {"ids": []}
    for v_ in vnames:
        merrors[v_] = []
        
    for idx_, stnid_ in enumerate(stnids):
        
        y_pred_ = model.predict(X[:, idx_, :, :])
        
        if len(y_pred_) >= 2:
            logging.debug("shape of y_pred[0].shape = {}, y_pred[1].shape = {}".format(y_pred_[0].shape, y_pred_[1].shape))
        else:
            logging.debug("shape of y_pred.shape = {}, y_pred = {}".format(y_pred_.shape, y_pred_))

        if task == "classification" or task == "class2step":
            if task == "class2step":
                y2step = y[:, idx_, :]
            y_class_pred = y_pred_
            y_class_true = y_class[:, idx_, :]
            y_clsout[:, idx_, :] = y_pred_
            
            logging.debug("shape of y_class_pred = {}, y_clsout[:, idx_, :] = {}".format(y_class_pred.shape, y_clsout[:, idx_, :].shape))
            logging.debug("shape of y_class_true = {}, y_class[:, idx_, :] = {}".format(y_class_true.shape, y_class[:, idx_, :].shape))
        else:
            if isinstance(y_, list) and len(y_) >= 2: 
                assert len(y_pred_) >= 2 
                y_pred = y_pred_[0]
                y_class_pred = y_pred_[1]
                y_class_true = y_class[:, idx_, :]
                y_clsout[:, idx_, :] = y_pred_[1]
            else:
                y_pred = y_pred_
            
            y_pred = scaler.inverse_transform(y_pred)
            y_true = y[:, idx_, :]

            logging.debug("shape of y_pred = {}, y_valout[:, idx_, :] = {}".format(y_pred.shape, y_valout[:, idx_, :].shape))
            logging.debug("shape of y_true = {}, y[:, idx_, :] = {}.".format(y_true.shape, y[:, idx_, :].shape))

            y_valout[:, idx_, :] = y_pred

        if outd is not None:

            XMissing = np.isnan(np.reshape(X[:, idx_, :, :], (-1, timestep * nfeature))).any(axis=1)
            
            if "y_true" in locals() and "y_pred" in locals():
                ypdf = pd.DataFrame(y_pred, columns=["{}_pred".format(vname) for vname in vnames])
                ytdf = pd.DataFrame(y_true, columns=["{}_true".format(vname) for vname in vnames])
                outdf = pd.concat([ytdf, ypdf], axis=1)
                outdf.index = datetimes
                outdf.to_csv("{}/pred/{}.csv".format(outd, stnid_))

                outnp = outdf.values
        #         errnp = np.apply_along_axis(lambda x: x[nfeature:] - x[0:nfeature], 1, outnp)
                errnp = np.apply_along_axis(lambda x: x[len(vnames):] - x[0:len(vnames)], 1, outnp)
                errv  = np.nanmean(errnp, axis=0)
                merrors["ids"].append(stnid_)
        #         merrors["errors"].append(errv)

                for vidx, v_ in enumerate(vnames):
                    merrors[v_].append(errv[vidx])

                datetimes_ = np.reshape(np.array(datetimes), (-1, 1))
                
#                 #  drop nan in y_pred 
#                 y_true = y_true[~np.isnan(y_pred).any(axis=1)]
#                 datetimes_ = datetimes_[~np.isnan(y_pred).any(axis=1)]
#                 if "y_class_pred" in locals() and "y_class_true" in locals():
#                     y_class_pred = y_class_pred[~np.isnan(y_pred).any(axis=1)]
#                     y_class_true = y_class_true[~np.isnan(y_pred).any(axis=1)]
#                 y_pred = y_pred[~np.isnan(y_pred).any(axis=1)]
              
#                 # drop nan in y_true
#                 y_pred = y_pred[~np.isnan(y_true).any(axis=1)]
#                 datetimes_ = datetimes_[~np.isnan(y_true).any(axis=1)]
#                 if "y_class_pred" in locals() and "y_class_true" in locals():
#                     y_class_pred = y_class_pred[~np.isnan(y_true).any(axis=1)]
#                     y_class_true = y_class_true[~np.isnan(y_true).any(axis=1)]
#                 y_true = y_true[~np.isnan(y_true).any(axis=1)]

                
                # drop missing 
        
                    
                logging.debug("shape of datetimes = {}, y_true = {}, y_pred = {} (id = {}, nsize = {})".format(datetimes_.shape, y_true.shape, y_pred.shape, stnid_, nsize))
                    
                    # onehot to integer
#                     y_class_pred = np.argmax(y_class_pred, axis=1)
#                     y_class_true = np.argmax(y_class_true, axis=1)
                
#                 datetimes_ = datetimes_.ravel().tolist()
            elif "y_class" in locals() and "y_class_pred" in locals():
                datetimes_ = np.reshape(np.array(datetimes), (-1, 1))

#                 y_class_true = y_class_true[~np.isnan(y_class_pred).any(axis=1)]
#                 datetimes_ = datetimes_[~np.isnan(y_class_pred).any(axis=1)]
#                 if task == "class2step":
#                     y2step = y2step[~np.isnan(y_class_pred).any(axis=1)]
#                 y_class_pred = y_class_pred[~np.isnan(y_class_pred).any(axis=1)]               
                
#                 y_class_pred = y_class_pred[~np.isnan(y_class_true).any(axis=1)]
#                 datetimes_ = datetimes_[~np.isnan(y_class_true).any(axis=1)]
#                 if task == "class2step":
#                     y2step = y2step[~np.isnan(y_class_true).any(axis=1)]
#                 y_class_true = y_class_true[~np.isnan(y_class_true).any(axis=1)]                   
                    
                    
            datetimes_ = datetimes_[~XMissing]
            if "y_true" in locals() and "y_pred" in locals():
                y_pred = y_pred[~XMissing]
                y_true = y_true[~XMissing]
            if "y_class_pred" in locals() and "y_class_true" in locals():
                y_class_pred = y_class_pred[~XMissing]
                y_class_true = y_class_true[~XMissing]        
            if task == "class2step":
                y2step = y2step[~XMissing]

            if len(datetimes_) <= 0: 
                logging.warning("sample size of {} = {}".format(stnid_, len(datetimes_)))
                continue

#             datetimes_ = datetimes_.ravel()
            datetimes_ = datetimes_.ravel().tolist()
            seidx = dtimes2mnseidx(datetimes_, hsystem=hsystem, tscale=tscale)
            seidx = seidx[1]

            for vidx, vname in enumerate(vnames):
                for Ym in seidx.index.tolist():
                    idx1, idx2 = seidx.loc[Ym]
                    if idx2 - idx1 <= max(ndel, 5):
                        logging.debug("idx1 = {}, idx2 = {} for {} on {}".format(idx1, idx2, stnid_, Ym))
                        continue

                    sdtime_ = datetimes_[idx1]
                    edtime_ = datetimes_[idx2]
                    
                    if isinstance(sdtime_, datetime):
                        if tscale == "M":
                            sdtime_ = datetime.strftime(sdtime_, "%Y%m%d%H%M")
                        else:
                            sdtime_ = datetime.strftime(sdtime_, "%Y%m%d%H")

                    if isinstance(edtime_, datetime):
                        if tscale == "M":
                            edtime_ = datetime.strftime(edtime_, "%Y%m%d%H%M")
                        else:
                            edtime_ = datetime.strftime(edtime_, "%Y%m%d%H")

#                     title = "{}_{}_{}_{}".format(vname, stnid_, datetimes_[idx1], datetimes_[idx2])                            
                    title = "{}_{}_{}_{}".format(vname, stnid_, sdtime_, edtime_)
                        
                    xposi_ = np.arange(idx1, idx2 + 1)[0:-1:math.floor((idx2 - idx1 + 1) / 5)].tolist()
                    xlabel = [datetimes_[i] for i in xposi_]
                    xposi = np.arange(idx2 - idx1 + 1)[0:-1:math.floor((idx2 - idx1 + 1) / 5)].tolist()

                    if classify is not None:
                        kwarg = {"title": title, "xposi": xposi, "xlabel": xlabel, "classify": classify}
                        if "y_class_pred" in locals() and "y_class_true" in locals():
                            kwarg.update({"y_class_true": y_class_true[idx1:(idx2 + 1), :], "y_class_pred": y_class_pred[idx1:(idx2 + 1), :]})
                            
                        if task == "classification" or task == "class2step":  # for classification 
#                             kwarg.update({"task": "classification"})
                            kwarg.update({"task": task})
    
                        if task == "class2step":
                            kwarg.update({"y2step": y2step[idx1:(idx2 + 1), vidx]})
                    else:  
                        kwarg = {"title": title, "xposi": xposi, "xlabel": xlabel}
                    
                    if not os.path.exists("{}/perfeval/{}/{}".format(outd, vname, stnid_)):
                        os.makedirs("{}/perfeval/{}/{}".format(outd, vname, stnid_)) 
                    
                    logging.info("{}/perfeval/{}/{}".format(outd, vname, stnid_))
                    
                    if task == "classification" or task == "class2step":
#                         pf.tspredict(y_class_true[idx1:(idx2 + 1)], y_class_pred[idx1:(idx2 + 1)], outd="{}/perfeval/{}".format(outd, vname), **kwarg)

                        # onehot to integer
                        y_class_pred_ = np.argmax(y_class_pred, axis=1)
                        y_class_true_ = np.argmax(y_class_true, axis=1)
                
                        pf.tspredict(y_class_true_[idx1:(idx2 + 1)], y_class_pred_[idx1:(idx2 + 1)], outd="{}/perfeval/{}/{}".format(outd, vname, stnid_), **kwarg)
                    else:
                        pf.tspredict(y_true[idx1:(idx2 + 1), vidx], y_pred[idx1:(idx2 + 1), vidx], outd="{}/perfeval/{}/{}".format(outd, vname, stnid_), **kwarg)
        
#                     gc.collect()
        
        gc.collect()
            
    # return predictions
    if outd is not None:
        if task == "classification" or task == "class2step":
            for tidx, dtime in enumerate(datetimes):

                if tidx < ndel:
                    continue
                
#                 if np.isnan(y_clsout[tidx, :, :]).all():
#                     continue
                
                ytdf = pd.DataFrame(np.argmax(y_class[tidx, :, :], axis=1), columns=["{}".format(vname) for vname in vnames])
                ypdf = pd.DataFrame(np.argmax(y_clsout[tidx, :, :], axis=1), columns=["{}Est".format(vname) for vname in vnames])
            
#                 ytdf = pd.DataFrame(y_class[tidx, :, :], columns=["{}_{}".format(vname, i) for i in range(nclass)])
                yplogits = pd.DataFrame(y_clsout[tidx, :, :], columns=["{}Est_{}".format(vname, i) for i in range(nclass)])
                outdf = pd.concat([ytdf, ypdf, yplogits], axis=1)
                outdf.index = stnids
                
                if isinstance(dtime, datetime):
                    if tscale == "H":
                        dtime = datetime.strftime(dtime, "%Y%m%d%H")
                    elif tscale == "M":
                        dtime = datetime.strftime(dtime, "%Y%m%d%H%M")
                    
                outdf.to_csv("{}/pred/cls_{}.csv".format(outd, dtime))
        else:            
            for tidx, dtime in enumerate(datetimes):
                
                if tidx < ndel:
                    continue
                    
#                 if np.isnan(y_valout[tidx, :, :]).all(): 
#                     continue
    
                if isinstance(dtime, datetime):
                    if tscale == "H":
                        dtime = datetime.strftime(dtime, "%Y%m%d%H")
                    elif tscale == "M":
                        dtime = datetime.strftime(dtime, "%Y%m%d%H%M")
    
                print(dtime, type(dtime), y[tidx, :, :].shape, y_valout[tidx, :, :].shape)
                ytdf = pd.DataFrame(y[tidx, :, :], columns=["{}".format(vname) for vname in vnames])
                ypdf = pd.DataFrame(y_valout[tidx, :, :], columns=["{}Est".format(vname) for vname in vnames])
                outdf = pd.concat([ytdf, ypdf], axis=1)
                outdf.index = stnids
                outdf.to_csv("{}/pred/val_{}.csv".format(outd, dtime))
               
                if isinstance(y_, list) and len(y_) >= 2:
#                 for tidx, dtime in enumerate(datetimes):
                    
#                     if tidx < ndel:
#                         continue
                        
#                     if np.isnan(y_clsout[tidx, :, :]).all():
#                         continue

                    ytdf = pd.DataFrame(np.argmax(y_class[tidx, :, :], axis=1), columns=["{}".format(vname) for vname in vnames])
                    ypdf = pd.DataFrame(np.argmax(y_clsout[tidx, :, :], axis=1), columns=["{}Est".format(vname) for vname in vnames])
                    outdf = pd.concat([ytdf, ypdf], axis=1)
                    outdf.index = stnids
                    outdf.to_csv("{}/pred/cls_{}.csv".format(outd, dtime))
                
            merrors["ids"].append("total")
            for vidx, v_ in enumerate(vnames):
                merrors[v_].append(np.nanmean(np.array(merrors[v_])))
            merrors = pd.DataFrame(merrors)
            merrors.to_csv("{}/pred/{}.csv".format(outd, "error_check"), index=False)

    if task == "classification": 
        return y_clsout
    elif task == "class2step": 
        return y_clsout
    elif isinstance(y_, list) and len(y_) >= 2:
        return [y_valout, y_clsout]
    else:
        return y_valout


# # LSTM with 60 cells and io with 4 features
# - total parameters = 4 * ((4 * 60 + 60) + (60 * 60)) + (60 * 4 + 4)
# - parameters of input gates (with bias)  = 4 * 60 + 60 + 60 * 60
# - parameters of forget gates (with bias) = 4 * 60 + 60 + 60 * 60
# - parameters of output gates (with bias) = 4 * 60 + 60 + 60 * 60
# - parameters of cell states (with bias)  = 4 * 60 + 60 + 60 * 60
# - parameters of output layer (with bias) = 60 * 4 + 4

# In[ ]:


def main(mode, tperiod, gif, ind=None, npyd=None, npysuffix=None, evald=None, 
         db=True, sampler="rus", 
         dsrc="hrf", n_in=6, n_out=1, t2last=0, vrange=None, vinfo=None, classify=None, rescale="MinMax", generator=False, 
         mconf=None, epochs=100, batchsize=100, saved_model=None, custom_objects=None, mname="NN"):
    
    '''
        1. mode:
            - train
            - test
            - finetune
            
        2. mname: name for saved model (training stage) or subdirectory of output "{}/{}".format(evald, mname) (testing stage) 
        
        3. mconf: a dict, 
        ex. 
            mconf = {
                "name": "DNNLSTM", 
                "units": [10, 30, 10, 30], 
                "outactfn": ["tanh", "sigmoid"],
                "outshape": [4, 1],
                "loss": ["mae", "categorical_crossentropy"],
                "lossw": [1, 1],                                    (default is None)
                "metric": ["mae", "accuracy"],                      (optional)
                "dropouts": 0.25 or a list, 
                "activations": "relu" or a list,
                "earlystopper": True,                               (default is True) 
                "task": "classification" or "class2step"}           (optional)
            where task: "class2step" for classification test or prediction

        4. rescale:
            - if rescale is not None == True == "Standard", then scaler is StandardScaler
            - if rescale is not None == "MinMax", then scaler is MinMaxScaler

        5. vrange: to specify features and their range (input: X)

        6. vinfo: to specify what variables to evaluate losses and set their range for inverse transform from normalization (input: y_true, y_class)
        
        7. classify: [[idx1, idx2, ...], [[values for classifing], [], ...]]
            ex.
                [[0, 3], [[-5, 0, 5, 10, 15], [0, 10, 20, 30, 40]]]
                0 to Temp, 3 to Precp         
    '''
    
    ndel = n_in
    if n_out > 1:
        ndel = n_in + n_out - 1
    
    dg = dgenerator(ind=ind, gif=gif, npyd=npyd)  # creat a object
    if vrange is None:
        if dsrc == "mdf":
            vrange = {"Temp": [-20.0, 50.0],
                      "RH": [0.0, 1.0],
                      "Pres": [600.0, 1100.0],
                      "Precp": [0.0, 220.0]}

            dg.vrange = pd.DataFrame(vrange)
        else:
            vrange = {"Temp": [-20.0, 50.0],
                      "RH": [0, 100],
                      "Pres": [600.0, 1100.0],
                      "Precp": [0.0, 220.0]}
            dg.vrange = pd.DataFrame(vrange)
    else:
        dg.vrange = pd.DataFrame(vrange)
    
#     vinfo = pd.DataFrame(dg.vrange)
    _vinfo = dg.vrange

    if vinfo is None:
        vinfo = _vinfo
    else:
        vinfo = pd.DataFrame(vinfo)
    
    _vnames = _vinfo.columns.tolist()
    _vrange = _vinfo.values
    nfeature = len(_vnames)
           
    vnames = vinfo.columns.tolist()
    vmapping = [_vnames.index(i) - len(_vnames) for i in vnames]

#     logging.info("features   = {}".format(_vnames))
    
    for v in _vnames:
        logging.info("{0:10s} = [{1:7.2f}, {2:7.2f}]".format(v, _vinfo[v].tolist()[0], _vinfo[v].tolist()[1]))

    logging.info("vdependent = {}".format(vnames))
    logging.info("vmapping = {}".format(vmapping))

    
    if evald is not None:
        if mode == "test" and not os.path.exists(evald):
            os.makedirs(evald)
    
    if npyd is not None:
        fnpy = True
    else:  # load data from ind
        fnpy = False
        
    if dsrc == "hrf":
        tscale = "H"
        hsystem = "01-24"
        if npysuffix is None:
            npysuffix = mode 
            
        if mode == "test":  # dim(dataset[0]) = (nsize, nstn, (n_in + n_out) * nfeature)
            dataset = dg.hrfgenerator(tperiod, fnpy=fnpy, n_in=n_in, n_out=n_out, t2last=t2last, mode=npysuffix, rescale=rescale, reformat=True, vstack=False, classify=classify, generator=False)
        else:  # dim(dataset[0]) = (nsize * nstn, (n_in + n_out) * nfeature)
            dataset = dg.hrfgenerator(tperiod, fnpy=fnpy, n_in=n_in, n_out=n_out, t2last=t2last, mode=npysuffix, rescale=rescale, reformat=True, vstack=True, classify=classify, generator=False)
        
    elif dsrc == "mdf":
        tscale = "M"
        hsystem = "00-23"
        if npysuffix is None:
            npysuffix = mode 
            
        if mode == "test":
            dataset = dg.mdfgenerator(tperiod, fnpy=fnpy, n_in=n_in, n_out=n_out, t2last=t2last, mode=npysuffix, rescale=rescale, reformat=True, vstack=False, classify=classify, generator=False)
        else:
            dataset = dg.mdfgenerator(tperiod, fnpy=fnpy, n_in=n_in, n_out=n_out, t2last=t2last, mode=npysuffix, rescale=rescale, reformat=True, vstack=True, classify=classify, generator=False)
    else:
        logging.error("data source must be 'hrf' or 'mdf'!")
        
    logging.info("tscale = {}".format(tscale))
    logging.info("hsystem = {}".format(hsystem))
    logging.info("len(dataset) = {}".format(len(dataset)))

    datetimes = dataset[-4]
    stnids    = dataset[-3]
    nsize     = len(datetimes)
    nstn      = len(stnids)
    
    logging.info("shape of rescaled data (dataset[0]) = {}".format(dataset[0].shape))
    logging.info("# of datetimes (len(dataset[-4])) = {}".format(nsize))
    logging.info("# of stations (len(dataset[-3])) = {}".format(nstn))
        
    if classify is not None:
        logging.info("shape of class labels (dataset[1]) = {}".format(dataset[1].shape))
    
    if mode == "test":
        X = np.reshape(dataset[0][:, :, 0:-nfeature], (-1, nstn, ndel, nfeature))
        if classify is not None:
            nclass = len(classify[1][0]) + 1
            assert dataset[0].shape[0] == dataset[1].shape[0]
            y_class = dataset[1][:, :, classify[0][0]]  # only one variable classification, ex. 3 for precp, shape: [nsize, nstn]
            y_onehot = np.ndarray((nsize, nstn, nclass), dtype=np.int)
            y_onehot.fill(-999)
            for i in range(nstn):
                y_class_ = pd.DataFrame({"y_class": y_class[:, i]})
                y_onehot_ = pd.get_dummies(y_class_["y_class"]).values
                if y_onehot_.shape[1] != nclass:
                    y_onehot_ = pd.get_dummies(y_class_["y_class"]).T.reindex(np.arange(nclass).tolist()).fillna(0).T.values
                                                                  
                y_onehot[:, i, :] = y_onehot_
                    
                    
            if vinfo is None:
#                 y = [X[:, -nfeature:], y_onehot]
                y = [dataset[-2], y_onehot]
            else:
#                 y = [X[:, vmapping], y_onehot]
                y = [dataset[-2][:, :, vmapping], y_onehot]
                
            if mconf is not None: 
                if "task" in mconf.keys() and mconf["task"] == "classification":
                    y = y_onehot
                    logging.info("shape, X: {}, y: {}".format(X.shape, y.shape))
                
#             else:
#                 if vinfo is None:
# #                     y = [X[:, -nfeature:], y_onehot]
#                     y = [dataset[-2], y_onehot]
#                 else:
# #                     y = [X[:, vmapping], y_onehot]
#                     y = [dataset[-2][:, :, vmapping], y_onehot]

                logging.info("shape, X: {}, y: {}, {}".format(X.shape, y[0].shape, y[1].shape))
            print('test-165', y)

        else:
            y = dataset[-2]
            logging.debug("shape of y = {} in main().".format(y.shape))
            if vinfo is not None:
                y = dataset[-2][:, :, vmapping]
                logging.debug("shape of y = {} if vinfo is not None.".format(y.shape))
    else:
        
        if "lossw" not in mconf.keys():
            mconf["lossw"] = None

        if "earlystopper" not in mconf.keys():
            mconf["earlystopper"] = True

        if "dropout" not in mconf.keys():
            mconf["dropout"] = 0

        if "recurrent_dropout" not in mconf.keys():
            mconf["recurrent_dropout"] = 0
        
        if classify is not None:
            assert dataset[0].shape[0] == dataset[1].shape[0]
            nclass = len(classify[1][0]) + 1
            y_class = dataset[1][:, classify[0][0]]  # only one variable classification, ex. 3 for precp
                        
            if db:

    #             scaled = np.reshape(dataset[0], (-1, n_in + n_out, nfeature))  # [nsize, n_in + n_out, nfeature]
    #             nsize_ = scaled.shape[0]
    #             y_class = np.zeros(nsize_)
    #             logging.debug("shape of scaled: {}".format(scaled.shape))

    #             precp_ = scaled[:, :, 3]
    #             scaled0precp = precp_.min()

    #             rains = np.zeros((nsize_), dtype=np.int)
    ##             rains[np.where(dataset[0][:, -1] > 0)] = 1
    #             rains[np.where(dataset[0][:, -1] > scaled0precp)] = 1

    #             rains = np.reshape(rains, (-1, 1))


    ##             nminority = precp_[np.any(precp_ != 0, axis=1)].shape[0]
    #             nminority = precp_[np.any(precp_ != scaled0precp, axis=1)].shape[0]

    #             nmajority = nsize_ - nminority
    ##             y_class[np.any(precp_ != 0, axis=1)] = 1
    #             y_class[np.any(precp_ != scaled0precp, axis=1)] = 1
    #             logging.debug("rain : others = {} : {} = {} : {}".format(nminority, nmajority, nminority / nminority, nmajority / nminority))

                clscounter = Counter(y_class)
                for key_ in clscounter.keys(): 
                    logging.info("class (scaled): {}, nclass: {}".format(key_, clscounter[key_]))

    #             rus = RandomUnderSampler(random_state=0) 
    
                if sampler == "tl":
                    s = TomekLinks(sampling_strategy=[0])
                else:
                    s = RandomUnderSampler(sampling_strategy=ratio_multiplier) 
                
                X_resampled, y_resampled = s.fit_resample(dataset[0], y_class)  # dim(dataset[0]) = (nsize * nstn, (n_in + n_out) * 4)
                

    #             if classify is not None:
    #                 assert len(mconf["outactfn"]) == len(mconf["outshape"]) == 2
    #                 X_resampled, y_resampled = rus.fit_resample(np.hstack([dataset[0], rains]), y_class)  # dim(dataset[0]) = (nsize * nstn, (n_in + n_out) * 4)
    #             else:
    #                 X_resampled, y_resampled = rus.fit_resample(dataset[0], y_class)  # dim(dataset[0]) = (nsize * nstn, (n_in + n_out) * 4)


                y_class = pd.DataFrame({"y_class": y_resampled})
                y_onehot = pd.get_dummies(y_class["y_class"]).values
                if y_onehot.shape[1] != nclass:
                    y_onehot = pd.get_dummies(y_class["y_class"]).T.reindex(np.arange(nclass).tolist()).fillna(0).T.values
            

                clscounter = Counter(y_resampled)
                for key_ in clscounter.keys(): 
                    logging.info("class (resampled): {}, nclass: {}".format(key_, clscounter[key_]))
                    
                X = np.reshape(X_resampled[:, :-nfeature], (-1, ndel, nfeature))


    #             if len(mconf["outactfn"]) == len(mconf["outshape"]) == 2:
    #                 X = np.reshape(X_resampled[:, :-5], (-1, n_in, 4))

    #                 if vinfo is None:
    #                     y = [X_resampled[:, -5:-1], X_resampled[:, -1]]
    #                 else:
    #                     y = [X_resampled[:, vmapping], X_resampled[:, -1]]
    #             else:
    #                 X = np.reshape(X_resampled[:, :-nfeature], (-1, n_in, nfeature))
    #                 if vinfo is None:
    #                     y = X_resampled[:, -nfeature:]
    #                 else:
    #                     y = X_resampled[:, vmapping]
            else:
                X = np.reshape(dataset[0][:, :-nfeature], (-1, ndel, nfeature))
                X_resampled = X
                y_resampled = y_class


            y_class = pd.DataFrame({"y_class": y_resampled})
            y_onehot = pd.get_dummies(y_class["y_class"]).values
                    
            if "task" in mconf.keys() and mconf["task"] == "classification":
                y = y_onehot
                logging.info("shape, X: {}, y: {}".format(X.shape, y.shape))
            else:
                assert len(mconf["outactfn"]) == len(mconf["outshape"]) == 2
                if vinfo is None:
                    y = [X_resampled[:, -nfeature:], y_onehot]
                else:
                    y = [X_resampled[:, vmapping], y_onehot]

                logging.info("shape, X: {}, y: {}, {}".format(X.shape, y[0].shape, y[1].shape))

        else:
            ###### only regression 
            X = np.reshape(dataset[0][:, :-nfeature], (-1, ndel, nfeature))
            if vinfo is None:                    
                y = dataset[0][:, -nfeature:]
            else:
                y = dataset[0][:, vmapping]

            logging.info("shape, X: {}, y: {}".format(X.shape, y.shape))
            
    if mode == "train":
#         history = train(X, y, epochs, batchsize, mconf, loss=loss, name=mname, lossw=lossw, earlystopper=earlystopper)        
        history = train(X, y, epochs, batchsize, mconf, name=mname)
        return [history, X, y]
    elif mode == "test":
#         scaler = preprocessing.MinMaxScaler()
        if vinfo is None:
            scaler = dataset[-1]
        else:
            if rescale == "MinMax":
                scaler = preprocessing.MinMaxScaler()
                scaler.fit(vinfo.values)
            else:
                scaler = preprocessing.StandardScaler()
                scaler.fit(y)

#         assert evald is not None
        evald_ = evald
        if evald is not None:
            evald_ = "{}/{}".format(evald, mname)
            
#         if "task" in mconf.keys() and mconf["task"] == "classification":

        

        if mconf is not None:
            if "task" in mconf.keys():
                y_out = test(X, y, saved_model, vinfo, scaler, datetimes, stnids, outd=evald_, ndel=ndel, custom_objects=custom_objects, task=mconf["task"], hsystem=hsystem, tscale=tscale, classify=classify[1][0])
        else:
            y_out = test(X, y, saved_model, vinfo, scaler, datetimes, stnids, outd=evald_, ndel=ndel, custom_objects=custom_objects, hsystem=hsystem, tscale=tscale)

        return y_out
    elif mode == "finetune":
#         saved_model = "/home/yuzhe/DataScience/QC/model/lstm1_0154_0.009_0.008_202008071814.hdf5"
        assert saved_model is not None
        finetune(X, y, epochs, batchsize, saved_model, modeld="model", ckptd="ckpt")    
        
        


# # Model

# In[ ]:


class MArgs(object):
    def __init__(self, mode, tperiod, vname, mname, dsrc, dbalance=False, dscaler="MinMax", epochs=100, batchsize=1000, earlystop=False, gmem=8, dhold=False, loglv="INFO"):
        self.mode = mode
        self.tperiod = tperiod
        self.vname = vname
        self.mname = mname
        self.dsrc = dsrc
        self.dbalance = dbalance
        self.dscaler = dscaler
        self.epochs = epochs
        self.batchsize = batchsize
        self.earlystop = earlystop
        self.gmem = gmem
        self.dhold = dhold
        self.loglv = loglv


# In[ ]:


if __name__ == "__main__":

#    jnb = True    
    jnb = False

    FORMAT = '%(asctime)s, %(funcName)s-%(lineno)d-%(levelname)s: %(message)s'
    
    if jnb:
        mode = "test"
#         mode = "train"
#         tperiod = [202010010000, 202010072350]
#         tperiod = [1998010101, 2015123124]
        tperiod = [2016010101, 2019123124]
        vname = "Precp123"
#         mname = "stackedLSTM"
        mname = "bidirectionalLSTM"
#         mname = "CNN1D"
#         dsrc = "mdf"
        dsrc = "hrf"
        dbalance = True
        epochs = 500
        batchsize = 1000
        earlystop = False

        args = MArgs(mode, tperiod, vname, mname, dsrc=dsrc, dbalance=dbalance, epochs=epochs, batchsize=batchsize, earlystop=earlystop)
    else:
        argp = argparse.ArgumentParser()
        argp.add_argument("-m", "--mode", dest="mode", type=str, help="mode, train or test")
        argp.add_argument("-t", "--tperiod", dest="tperiod", type=int, nargs=2, help="two integers, start datetime and end datetime")
        argp.add_argument("--vname", dest="vname", type=str)
        argp.add_argument("--mname", dest="mname", default="stackedLSTM", type=str)
        argp.add_argument("--dsrc", dest="dsrc", type=str, help="data source, hrf or mdf")
        argp.add_argument("--dbalance", dest="dbalance", action="store_true", help="data balance")
        argp.add_argument("--dscaler", dest="dscaler", default="MinMax", type=str, help="data scaler")

        argp.add_argument("--epochs", dest="epochs", default=100, type=int)
        argp.add_argument("--batchsize", dest="batchsize", default=1000, type=int)
        argp.add_argument("--earlystop", dest="earlystop", action="store_true", help="activate earlystopper")

        argp.add_argument("--gmem", dest="gmem", default=8, type=int)    # use how many mem in a device
        argp.add_argument("--dhold", dest="dhold", action="store_true")  # occupy a device
        argp.add_argument("--loglv", dest="loglv", default="INFO", type=str)

        args = argp.parse_args()
 
#    sys.exit()

#### config
    if args.loglv == "DEBUG":
        logLevel = logging.DEBUG
    elif args.loglv == "INFO":
        logLevel = logging.INFO
    else:
        logLevel = logging.WARNING


    epochs = args.epochs
    batchsize = args.batchsize
    custom_objects = {"YZKError": YZKError, 
                      "WeightedBinaryCrossntropy": WeightedBinaryCrossntropy, 
                      "ChangeableLossw": ChangeableLossw,
                      "TPAttention": TPAttention, 
                      "TPCNN1D": TPCNN1D}
    
    tperiod = args.tperiod
    
    if args.dsrc == "hrf":
        ind    = "/NAS-129/users1/T1/DATA/YY/ORG/HR1"
        logd   = "/home/yuzhe/DataScience/QC/log"
        npyd   = "/home/yuzhe/DataScience/dataset"
        modeld = "./model/ready"
        evald  = "/NAS-129/users1/T1/DATA/RNN/QC"
        gif    = "/home/yuzhe/CODE/ProgramT1/GRDTools/SRC/RES/GI/1500_decode_stationlist_without_space.txt"
      
#        tperiod_train = [1998010101, 2015123124]
#        tperiod_test  = [2016010101, 2019123124]
        
        
        # train 
#         loss = YZKError(element_weight=[1 / 6., 1 / 6., 1 / 6., 1 / 2.], penalized=-1)
      
        regloss = YZKError(penalized=-1)
        regloss = "mae"
#         regloss = YZKError()
        
#        mname = "LSTM1_DB_YZKPMAECos"
#        mname = "DNNLSTM_DB_MAE_WBC15"    
#        mname = "DNNLSTM_DB_YZKMAECos_WBC7_1_drop0p2_actfntanh"
#        mconf = {"name": "DNNLSTM", "units": [10, 30, 10, 30], "dropouts": 0.25, "activations": "relu"}
#        mconf = {"name": "DNNLSTM", "units": [20, 20, 20, 60], "outactfn": ["tanh", "sigmoid"], "outshape": [4, 1], "dropouts": 0.2, "activations": "relu"}
     
    elif args.dsrc == "mdf":
        ind    = "/NAS-DS1515P/users1/realtimeQC/ftpdata"
        logd   = "/home/yuzhe/DataScience/QC/log"
        npyd   = "/home/yuzhe/DataScience/dataset"
        modeld = "./model/ready"
        evald  = "/NAS-129/users1/T1/DATA/RNN/QC"
        gif    = "/NAS-DS1515P/users1/T1/res/stations.txt"
        
#        tperiod_train = [201801010100, 202007312350]
#         tperiod_test  = [202008010100, 202008312350]
#        tperiod_test  = [202008010000, 202009152350]
        
    if not os.path.exists(logd):
        os.makedirs(logd)
  
    if not os.path.exists("{}/{}".format(logd, args.vname)):
        os.makedirs("{}/{}".format(logd, args.vname))

    if not os.path.exists(npyd):
        os.makedirs(npyd)
    
    #     logging.getLogger().setLevel(logLevel)
    logging.basicConfig(level=logLevel, filename="{0}/{2}/{1}_{2}_{3}_{4}.log".format(logd, args.mname, args.vname, args.mode, args.dsrc), filemode="w", format=FORMAT)
    
#### gpu setting
    import tensorflow as tf
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            if args.dhold:
                nMB  = 1024 * args.gmem  # for tf.config.experimental.VirtualDeviceConfiguration
                tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=nMB)])
            else:
                tf.config.experimental.set_memory_growth(gpus[0], enable=True)
        except RuntimeError as e:
            logging.warning(e)

###########################################                               
###### 1. training for historical qc ######
###########################################
    if args.dsrc == "hrf":
        
        ###############################
        ### 1.1 stackedLSTM for Temp
        ###############################        
        if args.vname == "Temp": 
            
            mname = "{}_MAE_{}_HourlyTemp".format(args.mname,  args.dscaler)
            # vrange = {"Temp": [-20.0, 50.0],
            #           "RH": [0.0, 1.0],
            #           "Pres": [600.0, 1100.0]}
            vrange = {"Temp": [-20.0, 50.0]}
            vinfo = {"Temp": [-20.0, 50.0]}
                
            if args.mode == "train":
                regloss = "mae"    
                dropout = 0
                recurrent_dropout = 0
                
                # best units = [64]
                mconf = {"name": args.mname, 
                         "units": [32, 64, 32], 
                         "outactfn": ["sigmoid"], 
                         "outshape": [len(vinfo)], 
                         "loss": [regloss], 
                         "metric": ["mae"], 
                         "earlystopper": args.earlystop, 
                         "dropout": dropout, 
                         "recurrent_dropout": recurrent_dropout}
            
            
                ret = main(mode="train", tperiod=tperiod, gif=gif, npyd=npyd, npysuffix="hourlyTemp", 
                           db=args.dbalance, 
                           dsrc="hrf", n_in=6, n_out=1, vrange=vrange, vinfo=vinfo, rescale=args.dscaler,
                           mconf=mconf, epochs=epochs, batchsize=batchsize, mname=mname)
            
            elif args.mode == "test":
                saved_model = "{}/{}.hdf5".format(modeld, mname)
                
#                 main(mode="test", tperiod=tperiod_test, gif=gif, npyd=npyd, 
#                      db=False, n_in=6, n_out=1, dsrc="hrf", epochs=epochs, batchsize=batchsize, 
#                      npysuffix="testMinMax", saved_model=saved_model, evald=evald, mname=mname, custom_objects=custom_objects, vinfo=vinfo, rescale="MinMax")
              
                ret = main(mode="test", tperiod=tperiod, gif=gif, ind=ind, npyd=npyd, npysuffix="hourlyTemp", evald=evald, 
                           dsrc="hrf", n_in=6, n_out=1, vrange=vrange, vinfo=vinfo, rescale=args.dscaler, 
                           saved_model=saved_model, custom_objects=custom_objects, mname=mname)
        
        ###############################          
        ### 1.2 stackedLSTM for Precp
        ###############################
        elif args.vname == "Precp":  
            # 2012010101 - 2015123124
            # INFO:root:main-141, class (scaled): 0, nclass: 6865225
            # INFO:root:main-141, class (scaled): 1, nclass: 471860
            # INFO:root:main-141, class (scaled): 2, nclass: 45640
            # INFO:root:main-141, class (scaled): 3, nclass: 25844
            # INFO:root:main-141, class (scaled): 4, nclass: 13799
         
            mname = "{}_CCE_{}_HourlyPrecp".format(args.mname, args.dscaler)
            vrange = {"Temp": [-20.0, 50.0],
                      "RH": [0, 100],
                      "Pres": [600.0, 1100.0],
                      "Precp": [0.0, 220.0]}
            vinfo = {"Precp": [0.0, 220.0]}
            classify = [[3], [[0.1, 5, 10, 20]]]
#             classify = [[3], [[0.1]]]

            if args.mode == "train":
                
                regloss = "mae"
#                 clsloss = WeightedBinaryCrossntropy(element_weight=[1, 1])
                clsloss = "categorical_crossentropy"
#                 clsloss = "sparse_categorical_crossentropy"
#                 loss = [regloss, clsloss]
                lossw = [1, 3]
                metric = [tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None)]
                dropout = 0.
                recurrent_dropout = 0.
                
                task = "classification"
                
#                 mconf = {"name": "stackedLSTM", "units": [64, 64], "outactfn": ["sigmoid", "softmax"], "outshape": [1, 5], "loss": loss}
                mconf = {"name": args.mname, 
                         "units": [32, 64], 
                         "outactfn": ["softmax"], 
                         "outshape": [len(classify[1][0]) + 1], 
                         "loss": [clsloss], 
                         "metric": metric, 
                         "earlystopper": args.earlystop, 
                         "task": task, 
                         "dropout": dropout, 
                         "recurrent_dropout": recurrent_dropout}
          
                #ret = main(mode="train", tperiod=tperiod_train, gif=gif, npyd=npyd, mconf=mconf, 
                #           db=True, n_in=6, n_out=1, dsrc="hrf", epochs=epochs, batchsize=batchsize, 
                #           npysuffix="trainMinMax", mname=mname, vinfo=vinfo, classify=classify)
            
                ret = main(mode="train", tperiod=tperiod, gif=gif, npyd=npyd, npysuffix="hourly", 
                           db=args.dbalance, 
                           dsrc="hrf", n_in=6, n_out=1, vrange=vrange, vinfo=vinfo, rescale=args.dscaler, classify=classify,
                           mconf=mconf, epochs=epochs, batchsize=batchsize, mname=mname)
            elif args.mode == "test":
                saved_model = "{}/{}.hdf5".format(modeld, mname)
                task = "class2step"
#                 task = "classification"

                mconf = {"task": task}
                
#                ret = main(mode="test", tperiod=tperiod, gif=gif, npyd=npyd, 
#                           db=True, 
#                           dsrc="hrf", n_in=6, n_out=1, 
#                           npysuffix="testMinMax", saved_model=saved_model, evald=evald, mname=mname, custom_objects=custom_objects, vinfo=vinfo, rescale="MinMax")
                ret = main(mode="test", tperiod=tperiod, gif=gif, ind=ind, npyd=npyd, npysuffix="hourly", evald=evald, 
                           dsrc="hrf", n_in=6, n_out=1, vrange=vrange, vinfo=vinfo, rescale=args.dscaler, classify=classify,
                           saved_model=saved_model, mconf=mconf, custom_objects=custom_objects, mname=mname)
            
           
        ###############################################
        ### 1.3 CNN1D or bidirectionalLSTM for Precp
        ###############################################
        else:    

#             mname = "{}_MAEpCCE_{}".format(args.mname, args.dscaler)
            mname = "{}_CCE_{}".format(args.mname, args.dscaler)

            vrange = {"Temp": [-20.0, 50.0],
                      "RH": [0, 100],
                      "Pres": [600.0, 1100.0],
                      "Precp": [0.0, 220.0]}
            vinfo = {"Precp": [0., 220.]}
            classify = [[3], [[0.1, 5, 10, 20]]]

            if args.mode == "train":
#                 regloss = "mae"
                clsloss = "categorical_crossentropy"
#                 clsloss = WeightedBinaryCrossntropy(element_weight=[5, 1])
#                 loss = [regloss, clsloss]
                loss = [clsloss]
                lossw = [10.0, 0.1]
#                 metric = ["mae", tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None)]
                metric = [tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None)]

                dropout = 0.
                recurrent_dropout = 0.
                task = "classification"
                changeable_lossw = [1.5, 1.0]
        
                mconf = {"name": args.mname, 
                         "units": [16, 32], 
#                          "outactfn": ["sigmoid", "softmax"],
                         "outactfn": ["softmax"],                          
#                          "outshape": [len(vinfo), len(classify[1][0]) + 1], 
                         "outshape": [len(classify[1][0]) + 1], 
                         "activations": "relu",
#                          "loss": [regloss, clsloss],
                         "loss": [clsloss],
#                          "lossw": lossw,
                         "metric": metric, 
                         "earlystopper": args.earlystop, 
                         "task": task, 
                         "dropout": dropout, 
                         "recurrent_dropout": recurrent_dropout}
#                          "changeable_lossw": changeable_lossw}
          
                logging.info("mconf: {}".format(mconf))
                ret = main(mode="train", tperiod=tperiod, gif=gif, npyd=npyd, npysuffix="hourly", 
                           db=args.dbalance, 
                           dsrc="hrf", n_in=2, n_out=3, t2last=0, vrange=vrange, vinfo=vinfo, rescale=args.dscaler, classify=classify,
                           mconf=mconf, epochs=epochs, batchsize=batchsize, mname=mname)
            elif args.mode == "test":
                saved_model = "{}/{}.hdf5".format(modeld, mname)
                task = "class2step"

                mconf = {"task": task}
                ret = main(mode="test", tperiod=tperiod, gif=gif, npyd=npyd, npysuffix="hourly", evald=evald, 
                           dsrc="hrf", n_in=2, n_out=3, t2last=0, vrange=vrange, vinfo=vinfo, rescale=args.dscaler, classify=classify,
#                            saved_model=saved_model, custom_objects=custom_objects, mname=mname)
                           saved_model=saved_model, mconf=mconf, custom_objects=custom_objects, mname=mname)
    
#########################################
###### 2. training for realtime qc ######
#########################################
    elif args.dsrc == "mdf":
        
        ###################################################################
        ### 2.1 stackedLSTM for Temp 10min, dropout will break the model
        ###################################################################
        if args.vname == "Temp":
#            mname = "{}_MAE_MinMaxScaler_Temp10min".format(args.mname)
            mname = "{}_MAE_{}_Temp10min".format(args.mname, args.dscaler)
#             vrange = {"Temp": [-20.0, 50.0],
#                       "RH": [0.0, 1.0],
#                       "Pres": [600.0, 1100.0]}
#             vinfo = {"Temp": [-20.0, 50.0],
#                       "RH": [0.0, 1.0],
#                       "Pres": [600.0, 1100.0]}
            vrange = {"Temp": [-20.0, 50.0]}
            vinfo = {"Temp": [-20.0, 50.0]}
                
            if args.mode == "train":
                regloss = "mae"
#                 regloss = YZKError()
#                 dropout = 0.2
                dropout = 0
                recurrent_dropout = 0
              
                # best units = [64]
                mconf = {"name": args.mname, 
                         "units": [32, 64, 32], 
                         "outactfn": ["sigmoid"], 
                         "outshape": [len(vinfo)], 
                         "loss": [regloss], 
                         "metric": ["mae"], 
                         "earlystopper": args.earlystop, 
                         "dropout": dropout, 
                         "recurrent_dropout": recurrent_dropout}
            
                ret = main(mode="train", tperiod=tperiod, gif=gif, npyd=npyd, npysuffix="temp", 
                           db=args.dbalance, 
                           dsrc="mdf", n_in=6, n_out=1, vrange=vrange, vinfo=vinfo, rescale=args.dscaler,
                           mconf=mconf, epochs=epochs, batchsize=batchsize, mname=mname)
            elif args.mode == "test": 
                saved_model = "{}/{}.hdf5".format(modeld, mname)
                 
                ret = main(mode="test", tperiod=tperiod, gif=gif, ind=ind, npyd=None, npysuffix=None, evald=evald, 
                           dsrc="mdf", vrange=vrange, vinfo=vinfo, rescale=args.dscaler, 
                           saved_model=saved_model, custom_objects=custom_objects, mname=mname)
      
        ######################################
        ### 2.2 stackedLSTM for Precp 10min
        ######################################        
        elif args.vname == "Precp":
        
            mname = "{}_CCE_{}_Precp10min".format(args.mname, args.dscaler)
            vrange = {"Temp": [-20.0, 50.0],
                      "RH": [0.0, 1.0],
                      "Pres": [600.0, 1100.0], 
                      "Precp": [0.0, 220.0]}
            vinfo = {"Precp": [0.0, 220.0]}
            classify = [[3], [[0.1, 5, 10, 20]]]


            if args.mode == "train":
                # 201801010100 - 202007312350
                # INFO:root:main-141, class (scaled): 0, nclass: 30748053
                # INFO:root:main-141, class (scaled): 1, nclass: 2000755
                # INFO:root:main-141, class (scaled): 2, nclass: 205581
                # INFO:root:main-141, class (scaled): 3, nclass: 116764
                # INFO:root:main-141, class (scaled): 4, nclass: 56704
               
                clsloss = "categorical_crossentropy"
#                 clsloss = "sparse_categorical_crossentropy"
#                lossw = [1, 3]
                metric = tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None)
                dropout = 0
                recurrent_dropout = 0
                task = "classification"
              
                mconf = {"name": args.mname, 
                         "units": [32, 64, 32], 
                         "outactfn": ["softmax"], 
                         "outshape": [len(classify[1][0]) + 1], 
                         "loss": [clsloss], 
                         "metric": [metric], 
                         "earlystopper": args.earlystop,  # False 
                         "dropout": dropout, 
                         "recurrent_dropout": recurrent_dropout,
                         "task": task}
        
                ret = main(mode="train", tperiod=tperiod, gif=gif, npyd=npyd, npysuffix="train", 
                           db=args.dbalance, 
                           dsrc="mdf", n_in=6, n_out=1, vinfo=vinfo, rescale=args.dscaler, classify=classify,
                           mconf=mconf, epochs=epochs, batchsize=batchsize, mname=mname)
         
            elif args.mode == "test":
                # 3. loss: 0.1250 - categorical_accuracy: 0.9641 - val_loss: 4.3427 - val_categorical_accuracy: 0.6591 (recurrent_dropout = 0)
                # 2. loss: 0.1273 - categorical_accuracy: 0.9639 - val_loss: 4.3539 - val_categorical_accuracy: 0.6604
                # 1. loss: 0.1269 - categorical_accuracy: 0.9639 - val_loss: 4.2968 - val_categorical_accuracy: 0.6605
                saved_model = "{}/{}.hdf5".format(modeld, mname)
              
                task = "class2step"
                mconf = {"task": task}
              
                ret = main(mode="test", tperiod=tperiod, gif=gif, ind=ind, npyd=None, npysuffix=None, evald=evald, 
                           dsrc="mdf", vinfo=vinfo, rescale=args.dscaler, classify=classify, 
                           saved_model=saved_model, mconf=mconf, custom_objects=custom_objects, mname=mname)


    sys.exit()


