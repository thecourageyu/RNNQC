#!/usr/local/bin/python

import copy
import getpass
import math
import os
#import pymssql
import re
import sys
import time
import numpy as np
from datetime import datetime, timedelta


# =============
#   functions 
# =============
#def DateSplit(DateObj, DateType=0): 
def datetime_split(datetime_obj, hour_system=0):
 
    # datetime_obj is a datetime or str object, YYYYmmddHH
    # hour_system=0 for HH=00, 01, ..., 23
    # hour_system=1 for HH=01, 02, ..., 24        

    if isinstance(datetime_obj, str) or isinstance(datetime_obj, int):
        datetime_obj = str(datetime_obj)
        string_length = len(datetime_obj)
       
        mm = 0
        dd = 0
        HH = 0         
        if string_length == 10:
            YYYY = int(datetime_obj[0:4])
            mm = int(datetime_obj[4:6])
            dd = int(datetime_obj[6:8])
            HH = int(datetime_obj[8:10])
        elif string_length == 8:
            YYYY = int(datetime_obj[0:4])
            mm = int(datetime_obj[4:6])
            dd = int(datetime_obj[6:8])
        elif string_length == 6:
            YYYY = int(datetime_obj[0:4])
            mm = int(datetime_obj[4:6])

        return_obj = [YYYY, mm, dd, HH]
    else:
        time_tuple = datetime_obj.timetuple()
        return_obj = [time_tuple[0], time_tuple[1], time_tuple[2], time_tuple[3]]
    
    return return_obj


def available_cpu_count():
    """ Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program""" 

    # cpuset may restrict the number of "available" processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass    

    # Python 2.6+
    try:
        #import multiprocessing
        return mp.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # http://code.google.com/p/psutil/
    try:
        import psutil
        return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass


def chk_leap_year(year):
    #year = int(input("Enter a year: "))
    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                print("{0} is a leap year".format(year))
            else:
                print("{0} is not a leap year".format(year))
        else:
            print("{0} is a leap year".format(year))
    else:
        print("{0} is not a leap year".format(year))


def days_in_year(year):
    year = int(year)
    days = 0
    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                days = 366
            else:
                days = 365
        else:
            days = 366
    else:
        days = 365
    return(days)


def days_in_mon(mon, leap): # number of days for each month
    num4leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    num = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if leap == 1:
       days = num4leap[mon - 1]
    else:
       days = num[mon - 1]
    return days


def day_in_year(YYYYmmddHH):
    YYYYmmddHH = float(YYYYmmddHH)
    YYYY = int(math.floor(YYYYmmddHH / 10**6))
    mm = int(math.floor(YYYYmmddHH / 10**4)) - YYYY * 100
    dd = int(math.floor(YYYYmmddHH / 10**2)) - (YYYY * 10000 + mm * 100)
    HH = int(YYYYmmddHH) - (YYYY * 1000000 + mm * 10000 + dd * 100) # 01 ~ 24
    HH_idx = HH - 1 # 00 ~ 23
    dt = datetime(YYYY, mm, dd) # 0 <= hour argument <= 23, it's different of hr data
    opt = [YYYY, mm, dd, HH, dt.timetuple()[7]]
    return opt


def mk_not_exist_dir(path):
    try:
        if not os.path.exists(path):
             os.makedirs(path)
    except OSError as tmp:
        print(tmp)


def change_sysconfig(path, filename, user=None, pwd=None):

    sysconfig = []
    with open(path + "/" + filename, "r") as fid:
        for idx, tmp in enumerate(fid.readlines()):
            sysconfig.append(tmp.strip())

    #for loop in range(len(sysconfig)):
        #if loop == 5 - 1:
            #if user is not None:
                #sysconfig[loop] = "user = " + user
        #elif loop == 6 - 1:
            #if pwd is not None:
                #sysconfig[loop] = "password = " + pwd

    user_pattern = re.compile(pattern=r"^user.{0,2}=")
    pwd_pattern = re.compile(pattern=r"^password.{0,2}=")
    for idx, tmp in enumerate(sysconfig):
        if user is not None and user_pattern.match(tmp) is not None:
            sysconfig[idx] = "user = {0}".format(user)

        if pwd is not None and pwd_pattern.match(tmp) is not None:
            sysconfig[idx] = "password = {0}".format(pwd)

    with open(path + "/" + filename, "w") as fid:
        for idx, tmp in enumerate(sysconfig):
            fid.write(tmp + "\n")


def change_option(path, filename, sdt=None, edt=None):

    tmp_list = []
    with open(path + "/" + filename, "r") as fid:
        for idx, tmp in enumerate(fid.readlines()):
            tmp_list.append(tmp.strip())

    #for loop in range(len(tmp_list)):
        #if loop == 10 - 1:
            #if sdt is not None:
                #tmp_list[loop] = sdt
        #elif loop == 15 - 1:
            #if edt is not None:
                #tmp_list[loop] = edt

    row_counter = 0  
    for idx, tmp in enumerate(tmp_list):
        if len(tmp) <= 0: continue
        if tmp[0] != "#":
            row_counter = row_counter + 1
            if row_counter == 1 and sdt is not None:
                tmp_list[idx] = sdt
            elif row_counter == 2 and edt is not None:
                tmp_list[idx] = edt

    with open(path + "/" + filename, "w") as fid:
        for idx, tmp in enumerate(tmp_list):
            fid.write(tmp + "\n")


def getmmdd(irec, iyrlen):
  
    if (not (iyrlen==366 or iyrlen==365)):
      print("the second argument is not 366 and not 365...")
      sys.exit()
  
    mon_length_initial = [31,28,31,30,31,30,31,31,30,31,30,31]
    Elemon = np.ndarray(shape=(13), dtype=np.int_)
    Elemon.fill(0)
  
    mon_length = []
    for im in range(12):
        mon_length.append(mon_length_initial[im])
  
    if (iyrlen==365): # Feb
        mon_length[1] = mon_length_initial[1]
    elif (iyrlen==366):
        mon_length[1] = mon_length_initial[1] + 1
  
    SUMmon = 0    
    for im in range(12):
        SUMmon = SUMmon + mon_length[im]
        Elemon[im+1] = int(SUMmon)
  
    for im in range(12):
        if(Elemon[im]<=irec and irec<=Elemon[im+1]):
            mm = im + 1
            dd = irec - Elemon[im]
            break
     
    return([mm, dd])


def get_pentad(dtime, hour_system):
   # dtime is a datetime or str object
   # hour_system=0 for HH=00, 01, ..., 23
   # hour_system=1 for HH=01, 02, ..., 24

   pentad_matrix = np.ndarray(shape=(73, 2), dtype=np.int)
   pentad_matrix[ 0, 0] =  101; pentad_matrix[ 0, 1] =  105
   pentad_matrix[ 1, 0] =  106; pentad_matrix[ 1, 1] =  110
   pentad_matrix[ 2, 0] =  111; pentad_matrix[ 2, 1] =  115
   pentad_matrix[ 3, 0] =  116; pentad_matrix[ 3, 1] =  120
   pentad_matrix[ 4, 0] =  121; pentad_matrix[ 4, 1] =  125
   pentad_matrix[ 5, 0] =  126; pentad_matrix[ 5, 1] =  130
   pentad_matrix[ 6, 0] =  131; pentad_matrix[ 6, 1] =  204
   pentad_matrix[ 7, 0] =  205; pentad_matrix[ 7, 1] =  209
   pentad_matrix[ 8, 0] =  210; pentad_matrix[ 8, 1] =  214
   pentad_matrix[ 9, 0] =  215; pentad_matrix[ 9, 1] =  219
   pentad_matrix[10, 0] =  220; pentad_matrix[10, 1] =  224
   pentad_matrix[11, 0] =  225; pentad_matrix[11, 1] =  301
   pentad_matrix[12, 0] =  302; pentad_matrix[12, 1] =  306
   pentad_matrix[13, 0] =  307; pentad_matrix[13, 1] =  311
   pentad_matrix[14, 0] =  312; pentad_matrix[14, 1] =  316
   pentad_matrix[15, 0] =  317; pentad_matrix[15, 1] =  321
   pentad_matrix[16, 0] =  322; pentad_matrix[16, 1] =  326
   pentad_matrix[17, 0] =  327; pentad_matrix[17, 1] =  331
   pentad_matrix[18, 0] =  401; pentad_matrix[18, 1] =  405
   pentad_matrix[19, 0] =  406; pentad_matrix[19, 1] =  410
   pentad_matrix[20, 0] =  411; pentad_matrix[20, 1] =  415
   pentad_matrix[21, 0] =  416; pentad_matrix[21, 1] =  420
   pentad_matrix[22, 0] =  421; pentad_matrix[22, 1] =  425
   pentad_matrix[23, 0] =  426; pentad_matrix[23, 1] =  430
   pentad_matrix[24, 0] =  501; pentad_matrix[24, 1] =  505
   pentad_matrix[25, 0] =  506; pentad_matrix[25, 1] =  510
   pentad_matrix[26, 0] =  511; pentad_matrix[26, 1] =  515
   pentad_matrix[27, 0] =  516; pentad_matrix[27, 1] =  520
   pentad_matrix[28, 0] =  521; pentad_matrix[28, 1] =  525
   pentad_matrix[29, 0] =  526; pentad_matrix[29, 1] =  530
   pentad_matrix[30, 0] =  531; pentad_matrix[30, 1] =  604
   pentad_matrix[31, 0] =  605; pentad_matrix[31, 1] =  609
   pentad_matrix[32, 0] =  610; pentad_matrix[32, 1] =  614
   pentad_matrix[33, 0] =  615; pentad_matrix[33, 1] =  619
   pentad_matrix[34, 0] =  620; pentad_matrix[34, 1] =  624
   pentad_matrix[35, 0] =  625; pentad_matrix[35, 1] =  629
   pentad_matrix[36, 0] =  630; pentad_matrix[36, 1] =  704
   pentad_matrix[37, 0] =  705; pentad_matrix[37, 1] =  709
   pentad_matrix[38, 0] =  710; pentad_matrix[38, 1] =  714
   pentad_matrix[39, 0] =  715; pentad_matrix[39, 1] =  719
   pentad_matrix[40, 0] =  720; pentad_matrix[40, 1] =  724
   pentad_matrix[41, 0] =  725; pentad_matrix[41, 1] =  729
   pentad_matrix[42, 0] =  730; pentad_matrix[42, 1] =  803
   pentad_matrix[43, 0] =  804; pentad_matrix[43, 1] =  808
   pentad_matrix[44, 0] =  809; pentad_matrix[44, 1] =  813
   pentad_matrix[45, 0] =  814; pentad_matrix[45, 1] =  818
   pentad_matrix[46, 0] =  819; pentad_matrix[46, 1] =  823
   pentad_matrix[47, 0] =  824; pentad_matrix[47, 1] =  828
   pentad_matrix[48, 0] =  829; pentad_matrix[48, 1] =  902
   pentad_matrix[49, 0] =  903; pentad_matrix[49, 1] =  907
   pentad_matrix[50, 0] =  908; pentad_matrix[50, 1] =  912
   pentad_matrix[51, 0] =  913; pentad_matrix[51, 1] =  917
   pentad_matrix[52, 0] =  918; pentad_matrix[52, 1] =  922
   pentad_matrix[53, 0] =  923; pentad_matrix[53, 1] =  927
   pentad_matrix[54, 0] =  928; pentad_matrix[54, 1] = 1002
   pentad_matrix[55, 0] = 1003; pentad_matrix[55, 1] = 1007
   pentad_matrix[56, 0] = 1008; pentad_matrix[56, 1] = 1012
   pentad_matrix[57, 0] = 1013; pentad_matrix[57, 1] = 1017
   pentad_matrix[58, 0] = 1018; pentad_matrix[58, 1] = 1022
   pentad_matrix[59, 0] = 1023; pentad_matrix[59, 1] = 1027
   pentad_matrix[60, 0] = 1028; pentad_matrix[60, 1] = 1101
   pentad_matrix[61, 0] = 1102; pentad_matrix[61, 1] = 1106
   pentad_matrix[62, 0] = 1107; pentad_matrix[62, 1] = 1111
   pentad_matrix[63, 0] = 1112; pentad_matrix[63, 1] = 1116
   pentad_matrix[64, 0] = 1117; pentad_matrix[64, 1] = 1121
   pentad_matrix[65, 0] = 1122; pentad_matrix[65, 1] = 1126
   pentad_matrix[66, 0] = 1127; pentad_matrix[66, 1] = 1201
   pentad_matrix[67, 0] = 1202; pentad_matrix[67, 1] = 1206
   pentad_matrix[68, 0] = 1207; pentad_matrix[68, 1] = 1211
   pentad_matrix[69, 0] = 1212; pentad_matrix[69, 1] = 1216
   pentad_matrix[70, 0] = 1217; pentad_matrix[70, 1] = 1221
   pentad_matrix[71, 0] = 1222; pentad_matrix[71, 1] = 1226
   pentad_matrix[72, 0] = 1227; pentad_matrix[72, 1] = 1231

   #if isinstance(dtime, str):
   if isinstance(dtime, str) or isinstance(dtime, int):
       dtime = str(dtime)
       if len(dtime) > 8: 
           if dtime[8:10] == "24" and hour_system == 0:
               dtime = dtime[0:8] + "00"
               dtime = datetime.strptime(dtime, "%Y%m%d%H") + timedelta(days=1)
           elif dtime[8:10] == "24" and hour_system == 1:
               dtime = datetime.strptime(dtime[0:8], "%Y%m%d") 
           else:
               dtime = datetime.strptime(dtime[0:8], "%Y%m%d")
       else: 
           dtime = datetime.strptime(dtime, "%Y%m%d")

   dt_tuple = dtime.timetuple()
   mmdd = dt_tuple[1] * 10**2 + dt_tuple[2]
   for wpd in range(73):
       if pentad_matrix[wpd, 0] <= mmdd and mmdd <= pentad_matrix[wpd, 1]: 
           pentad_for_dt = wpd + 1
           break
   return(pentad_for_dt)   


def display_progress(idx, total, display_type=1):
#def display_progress(idx, total):
    # idx   : from 0 to total - 1
    # total : The number of proceeding tasks.

    # How many "=" will be displayed when 100 % is achieved.
    num_of_bar = 50

    # Should be inside a loop, total should be fixed! idx starts from 0
    
    #unit = total / 100.0
    #fprogress = " %3.2f"%((idx + 1) / unit)
    fprogress = " {0:3.2f}".format((idx + 1) / float(total) * 100.0)  # show xxx.xx % compeleted!
    done      = int((idx + 1) / float(total) * num_of_bar)            # completion rate
    not_yet   = num_of_bar - done

    #sys.stdout.write("\r    [" + "="*done + " "*not_yet + "]" + fprogress + "%")
    #sys.stdout.flush()

    if display_type == 1:
        sys.stdout.write("\r    [{0}{1}] {2}%".format('=' * done, ' ' * not_yet, fprogress))
        sys.stdout.flush()
    else:
        #sys.stdout.write("\r    [" + "="*done + " "*not_yet + "]" + fprogress + "%")
        sys.stdout.write("\r    [{0}{1}] {2}% ({3:>5d}/{4:>5d})".format('=' * done, ' ' * not_yet, fprogress, idx + 1, total))
        sys.stdout.flush()



def test_display_progress(total, tp):

    for i in range(total):
        time.sleep(0.0001)
        display_progress(i, total, tp)
    print("\n")

# ======================
#   get YYYYmmddHH0000
#   hour_type | 
#           1 | 1 ~ 24
#        o.w. | 0 ~ 23
# ======================
def YYYYmmddHH0000(start_year, end_year, hour_type):

    datetime_list = []
    start_datetime = datetime.strptime(str(start_year * 10**10 + 101000000), "%Y%m%d%H%M%S")
    end_datetime = datetime.strptime(str(end_year * 10**10 + 1231230000), "%Y%m%d%H%M%S")
    tmp_dt = start_datetime
    while tmp_dt <= end_datetime:
        tmp_str = tmp_dt.strftime("%Y%m%d%H%M%S")
        if hour_type == 1:
            tmp_HH = str(int(tmp_str[-6:-4]) + 1).zfill(2)
            tmp_str = tmp_str[0:-6] + tmp_HH + tmp_str[-4:]
        datetime_list.append(int(tmp_str))
        tmp_dt = tmp_dt + timedelta(hours=1) 
    return(datetime_list)

# ===================================================
#   return dates YYYYmmdd between start_year and end_year
# ===================================================
#def YYYYmmddList(start_year, end_year):
def YYYYmmdd(start_year, end_year):
    datetime_list = []
    start_datetime = datetime.strptime(str(start_year * 10**4 + 101), "%Y%m%d")
    end_datetime = datetime.strptime(str(end_year * 10**4 + 1231), "%Y%m%d")
    tmp_dt = start_datetime
    while tmp_dt <= end_datetime:
        tmp_str = tmp_dt.strftime("%Y%m%d")
        datetime_list.append(int(tmp_str))
        tmp_dt = tmp_dt + timedelta(days=1) 
    return(datetime_list)

def YYYYmm(start_year, end_year):
    datetime_list = []
    for tmp_yr in range(start_year, end_year + 1):
        for tmp_mm in range(1, 12 + 1):
            datetime_list.append(tmp_yr * 10**2 + tmp_mm) 
    return(datetime_list)


def ReadHRFormat(hrfile, **KeywordArgs):
    if os.path.isfile(hrfile):
        HRData = np.loadtxt(hrfile, dtype=np.str_, skiprows=1)
        StnID = HRData[0, 0]
        Datelist = HRData[0:, 1]
        #HRStr = 0
        #for idx, tmp in enumerate(Datelist):
            #HRStr = HRStr + 1
            #Datelist[idx] = Datelist[idx] + str(HRStr).zfill(2)
            #if HRStr == 24:
                #HRStr = 0
        DataMat = HRData[0:, 2:]
    else:
        print("Error: ", hrfile , " doesn't exist!")
        sys.exit(99) 

    return [StnID, Datelist, DataMat] 


def DerivedTD(TT, RH):
    # TT unit is K
    L = 2260 * 1000 # J/kg
    Rv = 461.5 # J/kg*K
    TD = -999
    minval = 0.00001
    
    if isinstance(TT, list) and isinstance(RH, list):
        if len(TT) == len(RH):
            TD = []
            nobs = len(TT)
            for loop in range(nobs):
                if TT[loop] == 0:
                    TT[loop] = minval
                TD.append((1 / TT[loop] - math.log(RH[loop]) * Rv / L ) ** (-1))
        else: 
            print("Error: length of two list are not equal!")
            sys.exit(99)
    #elif isinstance(TT, np.ndarray) and isinstance(RH, np.ndarray):
    else:    
        if TT == 0:
            TT = minval
        TD = (1 / TT - math.log(RH) * Rv / L ) ** (-1)
        #print "TD = {0},  TT = {1},  RH = {2}".format(TD - 273.15, TT - 273.15, RH)
        
    #if (TD - 273.15) > 30 or (TD - 273.15) < -30:
        #print "TD = {0},  TT = {1},  RH = {2}".format(TD - 273.15, TT - 273.15, RH)
    return(TD)





class BarometricFormula:
  def __init__(self,pres,pres_unit,temp,temp_unit):
    self.pres = pres
    self.pres_unit = pres_unit
    self.temp = temp
    self.temp_unit = temp_unit

  def __Barometeric__(self):
    self.pres 


###  packages: numpy, os 
class LoadText(): 
    def __init__(self, FilePath, FileName):
        self.FilePath = str(FilePath)
        self.FileName = str(FileName)

    def _FileExist(self, File):
        return os.path.exists(File)  
    
    def Load(self):
        File = self.FilePath.strip() + "/" + self.FileName.strip()
        print(File)
        if self._FileExist(File):
            DataArray = np.loadtxt(File, dtype=np.str_)
            print(DataArray)
            return DataArray
        else:
            print(" Error-Widgets-439: file {0} doesn't exist!".format(File))
            return -1


class DatetimeConvert:

    def __init__(self, tmpObj):
        self.tmpObj = tmpObj

    @staticmethod
    def TimeFMT(tmpObj):

        try:
            tmpObj = str(tmpObj)
        except Exception as ErrMsg:
            print(ErrMsg)
            return (-1)

        reObj = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}")
        if reObj.match(tmpObj) is not None:
            return tmpObj

        if len(tmpObj) == 6:
            tmpObj = tmpObj + "01000000"
        elif len(tmpObj) == 8:
            tmpObj = tmpObj + "000000"
        elif len(tmpObj) == 10:
            tmpObj = tmpObj + "0000"
        elif len(tmpObj) == 12:
            tmpObj = tmpObj + "00"
        else:
            print("Error: argument ", tmpObj, " can't identify.")
            return (-1)
        
        dtObj = datetime.strptime(tmpObj, "%Y%m%d%H%M%S")
        print(datetime.strftime(dtObj, "%Y-%m-%d %H:%M:%S"))
        return datetime.strftime(dtObj, "%Y-%m-%d %H:%M:%S")

    
    def ToDatetimeElement(self):
        
        tmpObj = self.tmpObj
        RTNObj = self.TimeFMT(tmpObj)
        if RTNObj is not (-1):
            dtObj = datetime.strptime(RTNObj, "%Y-%m-%d %H:%M:%S")
            return([dtObj, dtObj.timetuple()])
        else:
            return (-1)


    def ToDatetimeFMT(self):

        tmpObj = self.tmpObj
        if isinstance(tmpObj, int) or isinstance(tmpObj, str):
            print(tmpObj)
            return self.TimeFMT(tmpObj)
        elif isinstance(tmpObj, list):
            DatetimeFMTList = []
            for tmp in tmpObj:
                print(tmp)
                DatetimeFMTList.append(self.TimeFMT(tmp))
            return DatetimeFMTList
        else:
            print("Error: argument must be an integer, string or list object.")



