import os

from IutyLib.mutithread.threads import LoopThread
from IutyLib.commonutil.config import Config

from prx.PathProxy import PathProxy
from prx.CoachProxy import CoachProxy


ins_LoopThread = LoopThread(CoachProxy.doTrain)

class TrainProxy:
    
    def stop():
        rtn = {'success':False}
        print("s1")
        ins_LoopThread.stop()
        
        rtn['success'] = True
        return rtn
    
    def start(projectname,tag):
        rtn = {'success':False}
        print(ins_LoopThread)
        if ins_LoopThread._running:
            rtn['error'] = "System has in train process"
            return rtn
        
        if not CoachProxy.init(projectname,tag):
            rtn['error'] = "The coach can not start train process"
            return rtn
        ins_LoopThread = LoopThread(CoachProxy.doTrain)
        ins_LoopThread.start()
        
        rtn['success'] = True
        return rtn
        
    pass