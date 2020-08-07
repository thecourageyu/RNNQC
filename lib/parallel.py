#!/usr/bin/python3

# -*- coding: utf-8 -*-

"""
    Provides runFunctionsInParallel(), for managing a list of jobs (python functions) that are to be run in parallel.

    - Monitor progress of a bunch of function calls, running in parallel
    - Capture the output of each function call. This is a problem because Queues normally break if they get too full. Thus we regularly empty them.
    - Close the queues as functions finish. This is key because otherwise the OS shuts us down for using too many open files.

    Job results (returned values) should be pickleable.
"""

from datetime import datetime
from math import sqrt
from os import nice
from time import sleep

import gc  # Effort to close files (queues) when done...
import logging
import os
import sys
import time
import multiprocessing as mp

import numpy as np

__author__ = "Christopher Barrington-Leigh"

class pWrapper():  # Enclosing in a class originally motivated to make Garbage Collection work better
    def __init__(self, thefunc, theArgs=None, thekwargs=None, delay=None, name=None):
        self.callfunc   = thefunc
        self.callargs   = theArgs if theArgs is not None else []
        self.callkwargs = thekwargs if thekwargs is not None else {}
        # self.calldelay=delay  # Or should this be dealt with elsewhere?
        self.name       = name   # Or should this be dealt with elsewhere?
        self.funcName   = '(built-in function)' if not hasattr(self.callfunc, 'func_name') else self.callfunc.__name__
        self.gotQueue   = None   # collected output from the process
        self.started    = False
        self.running    = False
        self.finished   = False
        self.exitcode   = 'dns'  # "Did not start"
        self.is_alive   = 'dns'  # For internal use only. Present "running"
        self.queue      = 0      # Uninitiated queue is 0. Complete/closed queue will be None

    def get_func(self):
        return self.callfunc

    @staticmethod
    def add_to_queue(thefunc, que, theArgs=None, thekwargs=None, delay=None):
        """ This actually calls one job (function), with its arguments.
        To keep this method static, we avoid reference to class features """
        if delay:
            sleep(delay)
        #funcName  = '(built-in function)' if not hasattr(thefunc, 'func_name') else thefunc.__name__
        funcName  = thefunc.__name__
        theArgs   = theArgs if theArgs is not None else []
        kwargs    = thekwargs if thekwargs is not None else {}
        returnVal = que.put(thefunc(*theArgs, **kwargs))
        #print('pWrapper-60: Finished %s in parallel! ' % funcName)
        return(returnVal)  # this should be 0.

    def start(self):
        """ Create queue, add to it by calling add_to_queue """
        assert self.started == False
        self.queue  = mp.Queue()
        self.thejob = mp.Process(target=self.add_to_queue, 
                                 args=[self.callfunc, self.queue, self.callargs, self.callkwargs],)

        self.thejob.start()
        self.started = True
        self.running = True
        #print('pWrapper-71: Launching %s in parallel %s' %(self.funcName, self.name))

    def status(self): 
        """ 
            Get status of job, and empty the queue if there is something in it 

            Returns 
              - dns     (self.started is False)
              - 0       (self.finished is True)
              - failed  (self.finished is True)
              - running (self.finished is True)
        """
        if self.started is False:
            return('dns')
        if self.finished:
            return({0: '0', 1: 'failed'}.get(self.exitcode, self.exitcode))

        assert self.running

        self.is_alive = self.thejob.is_alive()
        cleanup = self.is_alive not in ['dns', 1]  # clean up if it is dead (not alive or not dns)

        assert self.running

        # Update/empty queue
        if not self.queue.empty():
            if self.gotQueue is None:  # the first queue get
                self.gotQueue = self.queue.get()
            else:
                self.gotQueue += self.queue.get()

        # Terminate the job, close the queue, try to initiate Garbage Collecting in order to avoid "Too many open files".
        # The following is intended to get around OSError: [Errno 24] Too many open files. But it does not. What more can I do to garbage clean the completed queues and jobs?
        if cleanup:
            self.cleanup()
        return('running')

    def cleanup(self):
        """ Attempts to free up memory and processes after one is finished, so OS doesn't run into problems to do with too many processes. """
        self.exitcode = self.thejob.exitcode  #  return exitcode as process finished
        self.thejob.join()
        self.thejob.terminate()
        self.queue.close()
        self.thejob = None
        self.queue = None
        self.finished = True
        self.running = False

    def queuestatus(self):
        """ 
            Check whether queue has overflowed 
            Returns
              - dns
              - 
              - empty
              - full
        """
        if self.queue in [0]:
            return('dns')  # Did not start yet
        if self.queue is None:
            return('')     # Closed
#        print('Queuestatus-pWrapper-114: empty'*self.queue.empty() + 'full'*self.queue.full())
        return('empty'*self.queue.empty() + 'full'*self.queue.full())


def runFunctionsInParallel(*args, **kwargs):
    """ 
        This is the main/only interface to class cRunFunctionsInParallel. See its documentation for arguments.
        It returns a tuple of two lists (success codes, return values).
    """
    if not args[0]:
        return ([], [])

    return cRunFunctionsInParallel(*args, **kwargs).launch_jobs()

###########################################################################################
###


class cRunFunctionsInParallel():
    """
        Run any list of functions, each with any arguments and keyword-arguments, in parallel.
        The functions/jobs should return (if anything) pickleable results. 
        In order to avoid processes getting stuck due to the output queues overflowing, the queues are regularly collected and emptied.

        You can now pass os.system or etc to this as the function, in order to parallelize at the OS level, with no need for a wrapper: I made use of hasattr(builtinfunction,'func_name') to check for a name.

        Parameters
        ----------
        
        FuncAndArgLists: a list of lists 
            List of up-to-three-element-lists, like [function, args, kwargs],
            specifying the set of functions to be launched in parallel.  
            If an element is just a function, rather than a list, then it is assumed
            to have no arguments or keyword arguments.
            Thus, possible formats for elements of the outer list are:
                - function
                - [function, list]
                - [function, list, dict]
        
        kwargs: dict
            One can also supply the kwargs once, for all jobs (or for those
            without their own non-empty kwargs specified in the list)
        
        names: an optional list of names to identify the processes.
            If omitted, the function name is used, so if all the functions are
            the same (ie merely with different arguments), then they would be
            named indistinguishably
        
        offsetsSeconds: int or list of ints
            delay some functions' start times
        
        allowJobFailure: True/False  [This parameter used to be called expectNonzeroExit)
            Normal behaviour is to not proceed if any function exits with a
            failed exit code. This can be used to override this behaviour.
        
        parallel: True/False
            Whenever the list of functions is longer than one, functions will
            be run in parallel unless this parameter is passed as False
        
        maxAtOnce: int
            If nonzero, this limits how many jobs will be allowed to run at once.  
            By default, this is set according to how many processors the hardware has available.
        
        showFinished : int or np.inf
            Specifies the maximum number of successfully finished jobs to show in the text interface 
            (before the last report, which should always show them all).
        
        Returns
        -------
        
        .launch_jobs() or .run() returns a tuple of (return codes, return values), each a list in order of the jobs provided.
        
        Issues
        -------
        
        Only tested on POSIX OSes.
        
        Examples
        --------
        
        See the testParallel() method in this module

    """

    def __init__(self, FuncAndArgLists, kwargs=None, names=None, parallel=None, offsetsSeconds=None, allowJobFailure=False, expectNonzeroExit=False, maxAtOnce=None, showFinished=20, monitor_progress=True):

        # Use parallel only when we have many processing cores (well, here, more than 8)
        self.parallel = mp.cpu_count() > 2 if parallel is None or parallel is True else parallel

        if not FuncAndArgLists:
            return  # list of functions to run was empty.

        if offsetsSeconds is None:
            offsetsSeconds = 0

        # faal: function as a list
        # Jobs may be passed as a function, not a list of [function, args, kwargs]:
        FuncAndArgLists = [faal if isinstance(faal, list) else [faal, [], {}] for faal in FuncAndArgLists]
        # Jobs may be passed with kwargs missing:
        FuncAndArgLists = [faal+[{}] if len(faal) == 2 else faal for faal in FuncAndArgLists]
        # Jobs may be passed with both args and kwargs missing:
        FuncAndArgLists = [faal+[[], {}] if len(faal) == 1 else faal for faal in FuncAndArgLists]
        # kwargs may be passed once to apply to all functions
        kwargs = kwargs if kwargs else [faal[2] for faal in FuncAndArgLists]

        if len(FuncAndArgLists) == 1:
            self.parallel = False

        if names is None:
            names = [None for f in FuncAndArgLists]
        self.names = [names[i] if names[i] is not None else f[0].__name__ for i, f in enumerate(FuncAndArgLists)]
        #self.funcNames = ['(built-in function)' if not hasattr(afunc, 'func_name') else afunc.__name__ for afunc, posi, kw in FuncAndArgLists]
        self.funcNames = [afunc.__name__ for afunc, posi, kw in FuncAndArgLists]

        assert len(self.names) == len(FuncAndArgLists)

        if maxAtOnce is None:
            self.maxAtOnce = max(1, mp.cpu_count() - 2)
        else:
            self.maxAtOnce = max(min(mp.cpu_count() - 2, maxAtOnce), 1)

        # For initial set of launched processes, stagger them with a spacing of the offsetSeconds.
        self.delays = list(((np.arange(len(FuncAndArgLists)) - 1) * (np.arange(len(FuncAndArgLists)) < self.maxAtOnce) + 1) * offsetsSeconds)
        self.offsetsSeconds = offsetsSeconds
        self.showFinished = showFinished
        if expectNonzeroExit:
            assert not allowJobFailure
            print('parallel.py: expectNonzeroExit is deprecated. Use (identical) allowJobFailure instead')
            allowJobFailure = expectNonzeroExit
        self.allowJobFailure = allowJobFailure

        nice(10)  # Add 10 to the niceness of this process (POSIX only)

        self.jobs        = None
        self.gotQueues   = dict()
        self.status      = [None for i, f in enumerate(FuncAndArgLists)]
        self.exitcodes   = [None for i, f in enumerate(FuncAndArgLists)]
        self.queuestatus = [None for i, f in enumerate(FuncAndArgLists)]

        self.FuncAndArgLists = FuncAndArgLists
        # If False, only report at the end.
        self.monitor_progress = monitor_progress

    def run(self):  # Just a shortcut
        return self.launch_jobs()

    def launch_jobs(self):

        if self.parallel is False:
            print('++++++++++++++++++++++  DOING FUNCTIONS SEQUENTIALLY ---------------- (parallel=False in runFunctionsInParallel)')
            returnVals = [fffargs[0](*(fffargs[1]), **(fffargs[2]))
                          for iffargs, fffargs in enumerate(self.FuncAndArgLists)]
            # In non-parallel a job failure should abort before returning. So we don't care about return values.
            return([0]*len(returnVals), returnVals)

        """ Use pWrapper class to set up and launch jobs and their queues. Issue reports at decreasing frequency. """
        self.jobs = [pWrapper(funcArgs[0], funcArgs[1], funcArgs[2], self.delays[i], self.names[i]) for i, funcArgs in enumerate(self.FuncAndArgLists)]
        # [Attempting to avoid running into system limits] 
        # Let's never create a loop variable which takes on the value of an element of the above list.
        # Always instead dereference the list using an index.
        # So no local variables take on the value of a job. (In addition, the job class is supposed to clean itself up when a job is done running).

        istart = self.maxAtOnce if self.maxAtOnce < len(self.jobs) else len(self.jobs)
        for ijob in range(istart):
            self.jobs[ijob].start()  # Launch them all (# of job = istart at once)

        timeElapsed = 0

        self.updateStatus()
        if 1:
            # This is not necessary; we can leave it to the first loop, below, to report. But for debug, this shows the initial batch.
            # 201807: ACtually, the first loop appears not to report. Reinstating this
            self.reportStatus(np.inf)
            #self.reportStatus(status, exitcodes, names, istart, showFinished)

        """ Now, wait for all the jobs to finish. Allow for everything to finish quickly, at the beginning. """
        lastreport = ''
        while any([self.status[i] == 'running' for i in range(len(self.jobs))]) or istart < len(self.jobs):
            sleepTime = 5 * (timeElapsed > 2)
            if timeElapsed > 0:
                # Wait a while before next update. Slow down updates for really long runs.
                time.sleep(1 + sleepTime)
            timeElapsed += sleepTime

            # Add any extra jobs needed to reach the maximum allowed:
            newjobs = 0
            while istart < len(self.jobs) and sum([self.status[i] in ['running'] for i in range(len(self.jobs))]) < self.maxAtOnce:
                self.jobs[istart].start()
                newjobs += 1
                self.updateStatus()
                if newjobs >= self.maxAtOnce:
                    lastreport = self.reportStatus(istart, previousReportString=lastreport)
                    newjobs = 0
                istart += 1
                timeElapsed = .01

            self.updateStatus()
            # self.reportStatus(status, exitcodes,names,istart,showFinished,  previousReportString=lastreport)
            lastreport = self.reportStatus(istart, previousReportString=lastreport)

        # All jobs are finished. Give final report of exit statuses
        self.updateStatus()
        self.reportStatus(np.inf)

        if any(self.exitcodes):
#            print('INPARALLEL: Parallel processing batch set did not ALL succeed successfully ('+' '.join(self.names)+')')
            print('INPARALLEL: Parallel processing batch set did not ALL succeed successfully.')
            # one of the functions you called failed.
            assert self.allowJobFailure
            print('          Tolerating job failure since allowJobFailure == True')
            return(False)
        else:
#            print('INPARALLEL: Apparent success of all functions (' + ' '.join(self.names)+ ')')
            print('INPARALLEL: Apparent success of all functions.')
        return (self.exitcodes, [self.gotQueues[idx] for idx in range(len(self.jobs))])

    def updateStatus(self):
        for idx in range(len(self.jobs)):
            if self.status[idx] not in ['failed', 'success', '0', '1', 0, 1]:                   #  dns, still runnig or status doesn't update 
                self.status[idx] = self.jobs[idx].status()            #  dns, 0, failed, running
                self.exitcodes[idx] = self.jobs[idx].exitcode         #  0, failed
                self.queuestatus[idx] = self.jobs[idx].queuestatus()  #  dns, , empty, full
            if self.status[idx] not in ['dns', 'running', None] and idx not in self.gotQueues:  # job is done (0 or failed) and doesn't get queue yet
                self.gotQueues[idx] = self.jobs[idx].gotQueue
                # jobs[idx].destroy()
                self.jobs[idx] = None
                gc.collect()

    def reportStatus(self, showmax, showsuccessful=None, previousReportString=''):

        if not self.monitor_progress:
            return('')

        if showsuccessful is None: showsuccessful = self.showFinished

        outs = ''

        ishowable = list(range(min(len(self.status), showmax)))
        istarted = [idx for idx in range(len(self.status)) if self.status[idx] not in ['dns']]
        isuccess = [idx for idx in ishowable if self.status[idx] in ['success', '0', 0]]
        irunning = [idx for idx in range(len(self.status)) if self.status[idx] in ['running']]

        earliestSuccess = -1 if len(isuccess) < showsuccessful else isuccess[::-1][showsuccessful - 1]  # the latest success?

        if 0:
            print(showmax, showsuccessful, earliestSuccess)
            print(len(isuccess) - showsuccessful)

        max_name_length = max([len(name) for name in self.names])
        max_funcname_length = max([len(name) for name in self.funcNames])
        sep_row = '-'*(max_name_length + 48 + max_funcname_length) + '\n'
        #table_fmt = '%' + str(max_name_length) + 's:\t%10s\t%10s\t%s()\n'
        table_fmt = '%{0}s:\t%10s\t%10s\t%s()\n'.format(max_name_length)
        outs += sep_row + table_fmt % ('Job', 'Status', 'Queue', 'Func') + sep_row 

        # Check that we aren't going to show more *successfully finished* jobs than we're allowed: Find index of nth-last successful one. 
        # That is, if the limit binds, we should show the latest N=showsuccessful ones only.
        outs += ''.join([table_fmt % (self.names[idx], self.status[idx], self.queuestatus[idx], self.funcNames[idx])
                         for idx in ishowable if self.status[idx] not in ['success', '0', 0] or idx >= earliestSuccess]) + '\n'

        # '' if self.jobs[idx] is None else '(built-in function)' if not hasattr(self.jobs[ii].get_func(),'func_name') else self.jobs[ii].get_func().func_name)

        # We don't hide failed jobs, but we do sometimes skip older successful jobs
        if len(isuccess) > showsuccessful:
            outs += '%d job%s running. %d other jobs finished successfully.\n' % (len(irunning), 's'*(len(irunning) != 1), len(isuccess) - showsuccessful)
        else:
            outs += '%d job%s running.\n' % (len(irunning), 's'*(len(irunning) != 1))

        if len(self.status) > len(istarted):
            outs += '%d more jobs waiting for their turn to start...\n' % (len(self.status) - len(istarted))
        
        #print('%d open queues...'%len(queues))
        outs += sep_row
        # return([exitcode(job) for ii,job in enumerate(sjobs)])
        if outs != previousReportString:
            print('\nReport status on {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            print(outs + '\n')

        return(outs)

    def emptyQueues(self):  # jobs,queues,gotQueues):
        for ii, job in enumerate(self.jobs):
            if ii not in self.queues or not isinstance(self.queues[ii], mp.queues.Queue):
                continue
            cleanup = self.exitcodes(job) == 0

            if not self.queues[ii].empty():
                if ii in gotQueues:
                    self.gotQueues[ii] += self.queues[ii].get()
                else:
                    self.gotQueues[ii] = self.queues[ii].get()
            # The following is intended to get arround OSError: [Errno 24] Too many open files.  But it does not. What more can I do to garbage clean the completed queues and jobs?
            if cleanup:
                job.join()
                job.terminate()
                self.queues[ii].close()
                """
        print('Joined job %d'%ii)
        job.terminate()
        print('Terminated job %d'%ii)
        queues[ii].close()
                """
                job = None
                #del job
                self.queues[ii] = None
                # del queues[ii] # This seems key. Before, when I kept queues in a list, deleting the item wasn't good enough.
                #print('                       Cleaning up/closing queue for job %d'%ii)


def breaktest():  # The following demonstrates how to clean up jobs and queues (the queues was key) to avoid the OSError of too many files open. But why does this not work, above? Because there's still a pointer in the list of queues? No,
    def dummy(inv, que):
        que.put(inv)
        return(0)
    from multiprocessing import Process, Queue
    nTest = 1800
    queues = [None for ii in range(nTest)]
    # [Process(target=dummy, args=[ii,queues[ii]]) for ii in range(nTest)]
    jobs = [None for ii in range(nTest)]
    # for ii,job in enumerate(jobs):
    for ii in range(nTest):  # ,job in enumerate(jobs):
        queues[ii] = Queue()
        job = Process(target=dummy, args=[ii, queues[ii]])
        job.start()
        print(('Started job %d' % ii))
        job.join()
        print(('Joined job %d' % ii))
        job.terminate()
        print(('Terminated job %d' % ii))
        queues[ii].close()
        queues[ii] = None  # This line does it!


def return_values():
    """ 
        DataFrames, for examples, don't have a boolean value. 
        What can we do to discern failures from return values?
    """


def test_allowJobFailure():  # In parallel==True case
    try:
        runFunctionsInParallel([[lambda: 1/0], [lambda: 1/0]])
        raise Exception
    except AssertionError:
        print(' ----> runFunctionsInParallel correctly objected to job failure.')
    try:
        runFunctionsInParallel(
            [[lambda: 1/0], [lambda: 1/0]], allowJobFailure=True)
        print(' ----> runFunctionsInParallel correctly tolerated job failure.')
    except AssertionError:
        raise Exception
    return


def testParallel():

    # Test display  of large number of jobs to check display settings
    def doodle3(jj, a=None, b=None, showFinished=12):
        i = 0
        while i<1e2:
            i = i+1
        return(jj)
    nTest = 2
#    runFunctionsInParallel([[doodle3, [ii], {'a': 5, 'b': 10}] for ii in range(nTest)], names=[str(
#        ii) for ii in range(nTest)], offsetsSeconds=0.2, maxAtOnce=40, parallel=True, allowJobFailure=True)

    
    
    # Demo longer jobs, since other demos' jobs finish too quickly on some platforms
    def doodle4():
        ll = np.random.randint(7)+3
        i = 0
        while i < 10**ll:
            i = i+1
        return(i)
    nTest = 2
#    runFunctionsInParallel([doodle4 for ii in range(nTest)], names=[str(
#        ii) for ii in range(nTest)], offsetsSeconds=None, maxAtOnce=10, showFinished=5)

    # Test use of kwargs
    def doodle1(jj, a=None, b=None):
        i = 0 + len(str(a))+len(str(b))
        while i < 1e2:
            i = i+1
        return(jj)
    nTest = 10
#    runFunctionsInParallel([[doodle1, [ii], {'a': 5, 'b': 10}] for ii in range(nTest)], names=[str(
#        ii) for ii in range(nTest)], offsetsSeconds=0.2, maxAtOnce=40, parallel=True, allowJobFailure=True)

    # Demo simpler use, function takes no arguments
    def doodle2():
        i = 0
        while i < 1e9:
            i = i+1
        return(i)
    nTest = 10
#    runFunctionsInParallel([doodle2 for ii in range(nTest)], names=[str(
#        ii) for ii in range(nTest)], offsetsSeconds=0.2, maxAtOnce=10, showFinished=5)

    # Test use of large number of jobs, enough to make some systems get upset without our countermeasures
    def doodle3(jj, a=None, b=None):
        a = np.ndarray((1000, 1000))
        i = 0
        while 0:  # i<1e2:
            i = i+1
        #return(jj)
        return(a)
    nTest = 27
#    a = runFunctionsInParallel([[doodle3, [ii], {'a': 5, 'b': 10}] for ii in range(nTest)], names=[str(
#        ii) for ii in range(nTest)], offsetsSeconds=0.2, maxAtOnce=40, parallel=True, allowJobFailure=True)

    indir = "/home/yuzhe/CODE/realtimeQC/ftpdata"
    pardir = "parsed"

    funs = [[qcfparser, [202002060510, [indir]], {"outdir": pardir}], 
            [qcfparser, [202002060520, [indir]], {"outdir": pardir}], 
            [qcfparser, [202002101020, [indir]], {"outdir": pardir}], 
            [qcfparser, [202002101120, [indir]], {"outdir": pardir}], 
            [qcfparser, [202002101230, [indir]], {"outdir": pardir}], 
            [qcfparser, [202002110720, [indir]], {"outdir": pardir}], 
            [qcfparser, [202002111150, [indir]], {"outdir": pardir}], 
            [qcfparser, [202002110500, [indir]], {"outdir": pardir}]] 
#    qcfparser("202002060510", ["/home/dpmit/YZK/realtime_qc/ftpdata"], outdir="parse")
    return_ = runFunctionsInParallel(funs, names=[str(ii) for ii in range(nTest)], 
                                     offsetsSeconds=0.2, maxAtOnce=40, parallel=True, allowJobFailure=True)

    print(return_[1][4])
    print(return_[1][0])
    print(return_[0])
    print(len(return_))

if __name__ == '__main__':
    sys.path.append("{0}/..".format(os.path.dirname(os.path.realpath(__file__))))
    from qc.load_data import Dataset
    qcfparser = Dataset.qcfparser

    testParallel()

