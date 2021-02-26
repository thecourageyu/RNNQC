#!/bin/bash

WORKDIR=/home/yuzhe/DataScience/QC

echo $WORKDIR
cd ${WORKDIR}

QCScript="./RNNQC.py"
sdtime=202011010100
edtime=202011302350
#vname=PrecpMAEpCCE
vname=Temp
#mname=CNN1D
mname=stackedLSTM
#dsrc=hrf
dsrc=mdf

TIME_STEP=6

# optional arguments parsed by getopts, : => need 1 argument after symbol ":"
while getopts "sd:" opts;
do
    case $opts in
        s)
            # set up datetime automatically
	    thisMonth=`date "+%Y%m"`

	    sdtime=`date -d "${thisMonth}01 00:00:00 1 hour ago" +"%Y%m%d%H%M"`
#            sdtime=${thisMonth}010000
            edtime=`date -d "-1 days" "+%Y%m%d"`2350
            echo dtime: ${sdtime} - ${edtime}
            ;;
        d)
            dtime=$OPTARG
            sdtime=`echo ${dtime} | cut -d" " -f1`
            edtime=`echo ${dtime} | cut -d" " -f2`
            if [ ${#edtime} -eq 0 ]; then
                edtime=${sdtime}
            fi

            echo dtime: ${sdtime} - ${edtime}
            ;;
        [?])
            echo non-optarg: $OPTARG
            ;;
    esac
done
#${QCScript} -m train -t ${sdtime} ${edtime} --vname ${vname} --mname ${mname} --dsrc ${dsrc} --dbalance --epochs 250

mname=stackedLSTM
dsrc=mdf

for vname in Precp Temp
do
    echo vname: ${vname}
    if [ ! -d log/${vname} ]; then
        mkdir -p log/${vname}
    fi

    ${QCScript} -m test -t ${sdtime} ${edtime} --vname ${vname} --mname ${mname} --dsrc ${dsrc} > log/${vname}/nohup.out 2>log/${vname}/nohup.err 
done 
