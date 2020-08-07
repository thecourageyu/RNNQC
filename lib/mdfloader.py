#!/usr/local/bin/python3

# -*- coding: utf-8 -*- 


#
#  2019-01-31, Fix queue block when put a numpy ndarray (ndarray can't be too large) into multiprocessing Queue 
#
#  Matainer: YZK
#  

import codecs
import copy
import ctypes
import logging
import math
import os
import pickle
import re
import shutil
import sys 
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

#wk_dir = os.path.dirname(os.path.realpath(__file__))
#os.chdir(wk_dir)

# this_file_dir =  os.path.dirname(os.path.realpath(__file__))
# sys.path.append("{0}/..".format(this_file_dir))
# import lib.widgets as wg

try:
    thisd = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(thisd)
    import widgets as wg
except Exception as E:
    logging.warning(E)
    if __name__ == "__main__":
        import widgets as wg
    else:
        import lib.widgets as wg


class Dataset:

    def __init__(self, 
                 bin_dir="EXEC", 
                 etc_dir="ROUTINE", 
                 log_dir="log", 
                 ref_dir="REF", 
                 input_dir="ftpdata", 
                 output_dir="output", 
                 stn_list="stalist.txt"):

#        print("Dataset class initialized with pid ", os.getpid())
        date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cpus = int(wg.available_cpu_count() * 0.5)
        if cpus >= 6: 
            cpus = 6
        self.cpus = cpus
                
        self.cwb_obs_file = ".QPESUMS_STATION.10M.mdf"
        self.auto_obs_file = ".QPESUMS_STATION.15M.mdf"
        self.parsing_dir = "parsing_data"
        self.parsing_file = ".QPESUMS_STATION.txt"
     
        self.bin_dir = bin_dir 
        self.etc_dir = etc_dir
        self.log_dir = log_dir
        self.ref_dir = ref_dir
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.stn_list = stn_list

        # make directories
        wg.mk_not_exist_dir(self.log_dir)
        wg.mk_not_exist_dir(self.output_dir)
        wg.mk_not_exist_dir(self.parsing_dir)

        self.load_data_log = open(self.log_dir + "/Dataset.log", "a")  
        for idx in range(100):
            self.load_data_log.write("=")
            if idx == 99:
                self.load_data_log.write("\n")
        self.load_data_log.write("*** datetime: " + date_now + "\n")

    @staticmethod
    def updateid(*parg):
    #def updateid(path, fname, stnid):
        
        if len(parg) == 3:
            fname = "{0}/{1}".format(parg[0].strip(), parg[1].strip())
            stnid = parg[2]
        else:
            fname = parg[0].strip()
            stnid = parg[1]
    
        contents = []
        #with open(path + "/" + fname, "r") as fid:
            #for idx, tmp in enumerate(fid.readlines()):
                #contents.append(tmp.strip())
    
        if isinstance(stnid, str):
            contents.append(stnid)
        elif isinstance(stnid, list):
            for tmp in stnid:
                contents.append(tmp)
    
        with open(fname, "w") as fid:
            for idx, tmp in enumerate(contents):
                fid.write("{0}\n".format(tmp))


    @staticmethod
    def floader(*parg):

        if len(parg) == 2:
            fname_ = "{0}/{1}".format(parg[0], parg[1])
        else:
            fname_ = parg[0]

        py_version = int(sys.version.split()[0].split(".")[0])
        #if py_version == 2:
            #fname = fname_
        #else: 
            #fname = codecs.open(fname_, encoding='big5')

        data_array = None
        if os.path.exists(fname_):
            fname = codecs.open(fname_, encoding='utf8')

            try:
                data_array = np.loadtxt(fname, dtype=np.str_)
            except Exception as E:
                logging.error("Dataset-floader-114: something went wrong\n {}!".format(E))

        return data_array


    @staticmethod
    def check_file_dir(file_name, file_dir):

        file_in_dir = None
        if isinstance(file_dir, list):
            for dir_name in file_dir:
                if os.path.exists("{0}/{1}".format(dir_name, file_name)):
                    file_in_dir = dir_name
        else:
            if os.path.exists("{0}/{1}".format(file_dir, file_name)):
                file_in_dir = file_dir

        return file_in_dir

    @staticmethod 
    def qcfparser(dtime, indirs, outdir=None, outfmt=".QPESUMS_STATION.txt", 
                  fmts=[".QPESUMS_STATION.10M.mdf", ".QPESUMS_STATION.15M.mdf", ".QPESUMS_GAUGE.10M.mdf"]):  
        '''
            order of fmts = [cwb, non-cwb, rain]
        ''' 

        py_version = int(sys.version.split()[0].split(".")[0])

        process_name = mp.current_process().name  
        #print("check subprocess name = {0}, pid = {1}".format(process_name, os.getpid()))

        # datetime_obj is a datetime object
        # dtime = datetime.strftime(dtobj, "%Y%m%d%H%M")
        dtime = str(dtime)
    
        if not isinstance(indirs, list):    
            indirs = [indirs]
   
        # data array includes LAT, LON, ELEV, TEMP, HUMD, PRES
        colnames1 = ["LAT", "LON", "ELEV", "TEMP", "HUMD", "PRES"]
        colnames2 = ["LAT", "LON", "ELEV", "RAIN"]
 
        ids = None  
        darray = None
        chnames = None
 
        fs = []  # file content 
        for fidx, fmt_ in enumerate(fmts):
 
            f_exist = False
            for indir in indirs: 
                fname_ = "{0}/{1}{2}".format(indir, dtime, fmt_)
                if os.path.exists(fname_):
                    f_exist = True
                    fname = fname_

#            print('qcfparser-185: ', fname, f_exist)
            if f_exist:
                f_open = codecs.open(fname, encoding="big5")
              
                try:
                    f = np.loadtxt(f_open, dtype=np.str_, skiprows=2)

                    fmt_correct = False
                    if fidx == 2:
                        header = np.copy(f[0, [3, 4, 5, 6]])
                        if False in [str(tmp) == colnames2[idx] for idx, tmp in enumerate(header)]: 
                            logging.error("Dataset-qcfparser-207: header is incorrect in {0}".format(fname))
                        else:
                            fmt_correct = True
                    else: 
                        header = np.copy(f[0, [3, 4, 5, 8, 9, 10]]) 
                        if False in [str(tmp) == colnames1[idx] for idx, tmp in enumerate(header)]: 
                            logging.error("Dataset-qcfparser-213: header is incorrect in {0}".format(fname))
                        else:
                            fmt_correct = True
                   
                    if fmt_correct:
                        fs.append(f)  
                    else:
                        fs.append(None) 
#                        cwb_data = np.copy(cwb_obs_record[1:, [4, 3, 5, 8, 9, 10]]) # notice lon and lat, yuzhe 20170723
#                        cwb_obs_alive = True
                except:
                    logging.error("Dataset-qcfparser-224: something went wrong when parsing {0}".format(fname))
                    fs.append(None)
            else:
                fs.append(None)

        if (fs[0] is not None) and (fs[1] is not None):
            ids = np.hstack((fs[0][1:, 0], fs[1][1:, 0]))
            chnames = np.hstack((fs[0][1:, 1], fs[1][1:, 1]))
            nrows = ids.shape[0]
            rain_ = np.zeros((nrows, 1))
            if (fs[2] is not None):
               rids = fs[2][1:, 0]
               rchnames = fs[2][1:, 1]
               rarray = fs[2][1:, [4, 3, 5, 6]] 
               for idx, tmp in enumerate(rarray):
                   if float(tmp[3]) > 0:
                       matched = np.where(ids == rids[idx])[0]
                       if len(matched) == 1:
                           rain_[matched[0]] = tmp[3]
                       elif len(matched) > 1:
                           logging.warning("Dataset-qcfparser-244: {0} shows up duplicately.".format(rids[idx])) 
                           
 
            darray = np.vstack((fs[0][1:, [4, 3, 5, 8, 9, 10]], fs[1][1:, [4, 3, 5, 8, 9, 10]]))
            darray = np.hstack((darray, rain_)).astype(dtype=np.float32)

        # output parsing file 

        if (fs[0] is not None) and (fs[1] is not None) and (outdir is not None):
            wg.mk_not_exist_dir(outdir)
            pout = "{0}/{1}{2}".format(outdir, dtime, outfmt)
            with open(pout, "w", encoding="utf8") as fid:
                for idx, tmp in enumerate(darray):
                    fid.write("{0:<6s} ".format(ids[idx]))
                    fid.write("{0:>11.4f}{1:>11.4f}{2:>10.1f}".format(tmp[0], tmp[1], tmp[2]))
                    fid.write("{0:>10.1f}{1:>10.2f}{2:>10.1f}{3:>10.1f}".format(tmp[3], tmp[4], tmp[5], tmp[6]))
                    #fid.write(" {0:<20s}\n".format(ch_name[idx].decode("big5").encode("utf-8")))
                    fid.write("  {0:<20s}\n".format(chnames[idx]))
       
        return [ids, darray, chnames, dtime]

    @staticmethod
    def qcrptloader(dtime, indir, fname, ftype="temp"):

        ids = None
        darray = None
        if ftype.strip() == "temp":
#            fheader = ['StnId', 'Temp', 'RH', 'Precp', 'Est_c', 'ErrStd', 'KErrStd', 'ErrCode', 'Est_1', 'Est_2', 'KStd_1', 'KStd_2', 'Sigma_1', 'Sigma_2', 'L1', 'L2', 'LEst_1', 'LEst_2', 'MaxDiff', 'DiffSTD', 'MaxBound']
            fheader = ['StnId', 'Temp', 'RH', 'Precp', 'Est_c', 'ErrStd', 'KErrStd',  'ECd']
            #fname_ = "{0}/bias_{1}.txt".format(indir, str(dtime))
        elif ftype.strip() == "pres":
            fheader = ['STNID', 'YYYYmmddHHMM', 'Pres_t', 'Pres_t-1', 'Pres_t-24', 'NStatus', 'lm_slope', 'PresTheory', 'TheoryBias', 'KrigingEst', 'EstBiasSTD', 'ErrorCode']

        fname_ = "{0}/{1}".format(indir, fname)
        if os.path.exists(fname_):
            try:
                f = np.loadtxt(fname_, dtype=np.str_)
                colnames = f[0, :] 
                
                for idx, name in enumerate(colnames):
                    if (idx + 1) > len(fheader): break
                    if not name.strip() == fheader[idx]:
                        logging.error('Dataset-qcrptloader-256: header of {} is incorrect!'.format(fname_))
                        return [ids, darray, dtime]
                
                ids = f[1:, 0]
                if ftype.strip() == "temp":
                    darray = f[1:, [1, 2, 3, 7]]
                elif ftype.strip() == "pres":
                    darray = f[1:, [2, 5, 9, 11]]

            except Exception as errmsg:
                 logging.error("Dataset-qcrptloader-296: load {} failed!\n {}".format(fname_, errmsg))
        else:
            logging.error("Dataset-qcrptloader-298: {} doesn't exist!".format(fname_))

        return [ids, darray, dtime] 

    @staticmethod
    def parsing_data(datetime_obj, 
                     qreturn=None, 
                     obs_dir="ftpdata",
                     other_obs_dirs=["ftpdata/PresQCed", "ftpdata/TxQCed", "ftpdata/QCed"], 
                     parsing_dir="parsing",
                     parsing_file=".QPESUMS_STATION.txt",
                     cwb_obs_file=".QPESUMS_STATION.10M.mdf", 
                     auto_obs_file=".QPESUMS_STATION.15M.mdf",
                     precp_obs_file=".QPESUMS_GAUGE.10M.mdf"): # obs_dir, parsing_dir
        
        # check version of python
        py_version = int(sys.version.split()[0].split(".")[0])
        
        process_name = mp.current_process().name  
        #print("check subprocess name = {0}, pid = {1}".format(process_name, os.getpid()))

        # datetime_obj is a datetime object
        datetime_str = datetime.strftime(datetime_obj, "%Y%m%d%H%M")
    

        # data array includes LAT, LON, ELEV, TEMP, HUMD, PRES
        var_list = ["LAT", "LON", "ELEV", "TEMP", "HUMD", "PRES"]
    
        obs_file = "{0}/{1}{2}".format(obs_dir, datetime_str, cwb_obs_file) #  fileencoding="big5"
        cwb_obs_alive = False
        try:
            if py_version == 2:
                obs_file_ = obs_file 
            else: 
                obs_file_ = codecs.open(obs_file, encoding='big5')
            cwb_obs_record = np.loadtxt(obs_file_, dtype=np.str_, skiprows=2)
            cwb_ch_name = np.copy(cwb_obs_record[1:, 1]) # chinese name of stations in cwb
            cwb_id = np.copy(cwb_obs_record[1:, 0]) # station id in cwb

            var_check = np.copy(cwb_obs_record[0, [3, 4, 5, 8, 9, 10]])
            if False in [str(tmp) == var_list[idx] for idx, tmp in enumerate(var_check)]: 
                print("IOError-Dataset-170: format is incorrect in file {0}\n".format(obs_file))
            else:   
                #cwb_data = np.copy(cwb_obs_record[1:, [0, 4, 3, 5, 8, 9, 10]]) # notice lon and lat, yuzhe 20170723
                cwb_data = np.copy(cwb_obs_record[1:, [4, 3, 5, 8, 9, 10]]) # notice lon and lat, yuzhe 20170723
                cwb_obs_alive = True
        except:
            if os.path.exists(obs_file):
                print("IOError-Dataset-177: something went wrong when parsing {0}\n".format(obs_file))
            else:
                print("IOError-Dataset-179: {0} doesn't exist!\n".format(obs_file))
        
        obs_file = obs_dir + "/" + datetime_str + auto_obs_file #  fileencoding="big5"
        auto_obs_alive = False
        try:
            if py_version == 2:
                obs_file_ = obs_file 
            else: 
                obs_file_ = codecs.open(obs_file, encoding='big5')
            auto_obs_record = np.loadtxt(obs_file_, dtype=np.str_, skiprows=2)
            auto_ch_name = np.copy(auto_obs_record[1:, 1])
            auto_id = np.copy(auto_obs_record[1:, 0])

            var_check = np.copy(auto_obs_record[0, [3, 4, 5, 8, 9, 10]])
            if False in [str(tmp) == var_list[idx] for idx, tmp in enumerate(var_check)]: 
                print("IOError-Dataset-194: format is incorrect in file {0}\n".format(obs_file))
            else: 
                #auto_data = np.copy(auto_obs_record[1:, [0, 4, 3, 5, 8, 9, 10]]) # notice lon and lat, yuzhe 20170723 
                auto_data = np.copy(auto_obs_record[1:, [4, 3, 5, 8, 9, 10]]) # notice lon and lat, yuzhe 20170723 
                auto_obs_alive = True
        except:
            if os.path.exists(obs_file):
                print("IOError-Dataset-201: something went wrong when parsing {0}\n".format(obs_file))
            else:
                print("IOError-Dataset-203: {0} doesn't exist!\n".format(obs_file))
    
        
        # load rainfall data
        rain_alive = False
        file_in_dir = Dataset.check_file_dir(datetime_str + precp_obs_file, other_obs_dirs)
        if file_in_dir is not None:

            obs_file = "{0}/{1}{2}".format(file_in_dir, datetime_str, precp_obs_file)           
            col_name = ["LAT", "LON", "ELEV", "RAIN"]
            try:
                if py_version == 2:
                    obs_file_ = obs_file 
                else: 
                    obs_file_ = codecs.open(obs_file, encoding='big5')
                rain_file = np.loadtxt(obs_file_, dtype=np.str_, skiprows=2)
                rain_ch_name = np.copy(rain_file[1:, 1])
                rain_id = np.copy(rain_file[1:, 0])

                var_check = np.copy(rain_file[0, [3, 4, 5, 6]])
                if False in [str(tmp) == col_name[idx] for idx, tmp in enumerate(var_check)]: 
                    print("IOError-Dataset-224: format is incorrect in file {0}\n".format(obs_file))
                else:            
                    #rain_data = np.copy(rain_file[1:, [0, 4, 3, 5, 6]]) # notice lon and lat, yuzhe 20170723 
                    rain_data = np.copy(rain_file[1:, [4, 3, 5, 6]]) # notice lon and lat, yuzhe 20170723 
                    rain_alive = True
            except:
                if os.path.exists(obs_file):
                    print("IOError-Dataset-231: something went wrong when parsing {0}\n".format(obs_file))
                else:
                    print("IOError-Dataset-233: {0} doesn't exist!\n".format(obs_file))

        if cwb_obs_alive and auto_obs_alive:
            ch_name = np.hstack((cwb_ch_name, auto_ch_name))
            stn_id = np.hstack((cwb_id, auto_id))
            
            num_of_rows = stn_id.shape[0]
            rain_data_ = np.zeros((num_of_rows, 1))
            if rain_alive:
                for tmp_idx, tmp_data in enumerate(rain_data):
                    if float(tmp_data[3]) > 0: 
                        match_idx = np.where(stn_id == rain_id[tmp_idx])[0]
                        if len(match_idx) == 1:
                            rain_data_[match_idx[0]] = tmp_data[3]
                        elif len(match_idx) > 1:
                            print("Warning-Dataset-248: {0} shows up duplicately.".format(rain_id[tmp_idx])) 

            data_array = np.vstack((cwb_data, auto_data))
            data_array = np.hstack((data_array, rain_data_)).astype(dtype=np.float32) # resize or change dtype

        else:
            ch_name = None
            stn_id = None
            data_array = None 
       
        # output parsing file 
        if (cwb_obs_alive and auto_obs_alive) and (qreturn is not None):
            parsing_output = "{0}/{1}{2}".format(parsing_dir, datetime_str, parsing_file)
            with open(parsing_output, "w") as parsing_opt:
                for idx, tmp in enumerate(data_array):
                    parsing_opt.write("{0:<6s} ".format(stn_id[idx]))
                    parsing_opt.write("{0:>11.4f}{1:>11.4f}{2:>10.1f}".format(tmp[0], tmp[1], tmp[2]))
                    parsing_opt.write("{0:>10.1f}{1:>10.2f}{2:>10.1f}{3:>10.1f}".format(tmp[3], tmp[4], tmp[5], tmp[6]))
                    #parsing_opt.write(" {0:<20s}\n".format(ch_name[idx].decode("big5").encode("utf-8")))
                    parsing_opt.write("  {0:<20s}\n".format(ch_name[idx]))
       
        if ch_name is not None:
            for idx, tmp in enumerate(ch_name):
                #ch_name[idx] = "{0:<20s}".format(tmp.decode("big5").encode("utf8"))
                #ch_name[idx] = "{0:<20s}".format(tmp)
                ch_name[idx] = tmp

        if qreturn is None:
            return [stn_id, data_array, ch_name]
        else:
            #print("check memory usage (bytes) = {0}".format(data_array.nbytes))
            qreturn.put([stn_id, data_array, ch_name, datetime_obj])
            #qreturn.put([stn_id, data_array, datetime_obj])
            #qreturn.put([datetime_obj, stn_id, pickle.dumps(data_array), ch_name])
            #qreturn.put(pickle.dumps(data_array))

    @staticmethod
    def getGI(fname):
        GI  = Dataset.floader(fname) #  file encoding = 'utf-8'
        ids  = []
        nstn = None
        if GI is not None:
            if len(GI.shape) == 1:
                nstn = int(1)
                ids.append(GI[0])
            else:
                nstn = int(GI.shape[0])
                for idx in range(nstn):
                    ids.append(GI[idx, 0])
        
        return [GI, ids, nstn]
 

    def load_data(self, start_point=201810010000, end_point=201812312350, num_of_lag=6, imputations=True):
#        print("Dataset class load_data with pid ", os.getpid())
     
        # value bounding interval
        temp_l_b = -7.

        # station list  
        stn_list, id_list, num_of_stn = self.load_stnlist("{0}/{1}".format(self.ref_dir, self.stn_list)) #  file encoding = 'utf-8'
        if snt_list is None: 
            self.load_data_log.write("IOError: {0}/{1} doesn't exists!\n".format(self.ref_dir, self.stn_list))
            self.load_data_log.close()
            for idx in range(100):
                self.load_data_log.write("=")
                if idx == 99:
                    self.load_data_log.write("\n\n")
            sys.exit(99)
        
        # get pressure 
        re_obj = re.compile(pattern=r"^([0-9]{12}).*")

        input_files = os.listdir(self.input_dir)
        input_datetime = [int(tmp.split(".")[0]) for tmp in input_files if re_obj.match(tmp) is not None]
#        input_files = [re_obj.match(tmp).group()[0] for tmp in input_files]
        input_datetime = np.unique(input_datetime)
        input_datetime = input_datetime[(start_point <= input_datetime) & (input_datetime <= end_point)]
        input_datetime = np.sort(input_datetime)
#        print(input_datetime)

        if len(input_datetime) > 0:

            start_datetime = datetime.strptime(str(start_point), "%Y%m%d%H%M")
            end_datetime = datetime.strptime(str(end_point), "%Y%m%d%H%M")
            tmp_datetime = start_datetime

            num_of_snip = 0
            while tmp_datetime <= end_datetime: 
                num_of_snip += 1
                tmp_datetime = tmp_datetime + timedelta(minutes=10)
            
            #sdate = datetime.strptime(str(MinDateInt), "%Y%m%d%H%M") - timedelta(days=14)
            #edate = datetime.strptime(str(MaxDateInt), "%Y%m%d%H%M")
            #syr = sdate.timetuple()[0]
            #eyr = edate.timetuple()[0]
            
            output_data_array = np.ndarray(shape=(num_of_snip, 6 + num_of_lag, num_of_stn)) # lon, lat, elev, temperature, humidity, pressure 
            output_data_array.fill(-999.)

            tmp_datetime = start_datetime
            dim_1_idx = 0
            counts = 0
            job_list = []
            queue_list = []
            datetime_list = []
            output_datetime = []
            #queue_obj = mp.Queue() 
            while tmp_datetime <= end_datetime:
                queue_list.append(mp.Queue())  
                datetime_list.append(tmp_datetime)
                output_datetime.append(tmp_datetime)
                counts += 1
                #parsing_process = mp.Process(target=self.parsing_data,
                                             #name=datetime_list[-1].strftime("%Y%m%d%H%M"), 
                                             #args=(datetime_list[-1], 
                                                   #queue_list[-1], ))
 
                parsing_process = mp.Process(target=self.parsing_data, 
                                             args=(datetime_list[-1], 
                                                   queue_list[-1], 
                                                   self.input_dir, 
                                                   self.parsing_dir, 
                                                   self.parsing_file, 
                                                   self.cwb_obs_file, 
                                                   self.auto_obs_file))

                job_list.append(parsing_process)
                job_list[-1].daemon = False

                tmp_datetime = tmp_datetime + timedelta(minutes=10)
                
                if counts == self.cpus or tmp_datetime == end_datetime: 
                    
                    [job.start() for job in job_list] 
                    [job.join() for job in job_list]
 
                    for que_idx in range(counts):
                        dim_1_idx += 1
                        if queue_list[que_idx].empty():
                            print("Warning-Dataset-387: queue is empty, dim1={0}, datetime={1}".format(dim_1_idx, datetime_list[que_idx]))
                            continue
                        que_obj = queue_list[que_idx].get()
                        que_datetime = datetime_list[que_idx]
                        #dateobj = que_obj[0].timetuple()
                        stn_id = que_obj[0] 
                        np_obj = que_obj[1] # it's a numpy ndarray
                        if np_obj is None: # To take obs from parsing_data in order to make a time series is continuous
                            print("Warning-Dataset-395: ndarray in queue is None, dim1={0}, datetime={1}".format(dim_1_idx, datetime_list[que_idx]))
                            continue
            
                        for row_idx, id_in_queue in enumerate(stn_id):
                            
                            if not (id_in_queue in id_list): 
                                continue
               
                            for id_idx, id_in_list in enumerate(id_list):
                                if id_in_list == id_in_queue:
                                    output_data_array[dim_1_idx, 0:6, id_idx] = np_obj[row_idx, 0:6]
                                    

                    counts = 0
                    job_list = []
                    queue_list = [] 
                    datetime_list = [] 

            # data imputations by sample mean (temperature)
            if imputations:
                fid = open(os.path.join(self.log_dir, "low_data_volume.txt"), "w")
                for id_idx, id_in_list in enumerate(id_list):

                    num_of_non_missing = np.where(np.ravel(output_data_array[:, 3, id_idx] > temp_l_b) == 1)[0].shape[0]
                    if float(num_of_non_missing) / float(num_of_snip) < 0.8:            
                        print("Warning-Dataset-420: low data volume for id = {0}".format(id_in_list))
                        fid.write("{0}\n".format(id_in_list))
                        continue
                    
                    output_data_array[:, 0, id_idx] = stn_list[id_idx, 1]
                    output_data_array[:, 1, id_idx] = stn_list[id_idx, 2]
                    output_data_array[:, 2, id_idx] = stn_list[id_idx, 3]

                    temp_mean = np.mean(output_data_array[:, 3, id_idx][output_data_array[:, 3, id_idx] >= temp_l_b])
                    output_data_array[:, 3, id_idx][output_data_array[:, 3, id_idx] < temp_l_b] = temp_mean

                    rh_mean = np.mean(output_data_array[:, 4, id_idx][output_data_array[:, 4, id_idx] >= 0.])            
                    output_data_array[:, 4, id_idx][output_data_array[:, 4, id_idx] < 0.] = rh_mean
                
                    pres_mean = np.mean(output_data_array[:, 5, id_idx][output_data_array[:, 5, id_idx] >= 0.])
                    output_data_array[:, 5, id_idx][output_data_array[:, 5, id_idx] < 0.] = pres_mean
                           
                    for lag_idx in range(num_of_lag):
#                        print(0, (lag_idx + 1), num_of_snip, num_of_snip - lag_idx - 1)
                        output_data_array[0:(lag_idx + 1), 6 + lag_idx, id_idx] = temp_mean
                        output_data_array[(lag_idx + 1):, 6 + lag_idx, id_idx] = output_data_array[0:(num_of_snip - lag_idx - 1), 3, id_idx] 

                fid.close()      

            columns = ["Lon", "Lat", "Elev", "Temp", "RH", "Pres"]
            for idx in range(num_of_lag):
                columns.append("Lag-{0}".format(idx + 1))

            for id_idx, id_in_list in enumerate(id_list):
                output_df = pd.DataFrame()
                output_df["datetime"] = output_datetime
                for col_idx in range(6 + num_of_lag):
                    output_df[columns[col_idx]] = output_data_array[:, col_idx, id_idx]
                output_df.to_csv("{0}/time_series_{1}.csv".format(self.output_dir, id_in_list), index=False, float_format="%.2f")
            
            for idx in range(100):
                self.load_data_log.write("=")
                if idx == 99:
                    self.load_data_log.write("\n\n")
        else: 
            datetime_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.load_data_log.write("\nWarning: There is nothing to check on !" + datetime_now + "\n") 
            for idx in range(100):
                self.load_data_log.write("=")
                if idx == 99:
                    self.load_data_log.write("\n\n")


#if __name__ == "__main__":
    
