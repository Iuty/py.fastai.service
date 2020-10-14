import os

from IutyLib.mutithread.threads import LoopThread
from IutyLib.commonutil.config import Config

from prx.PathProxy import PathProxy
from prx.CoachProxy import CoachProxy


class TrainProxy:
    """
    api here
    """
    def stop():
        rtn = {'success':False}
        
        ins_TrainProxy.stopTrain()
        
        rtn['success'] = True
        return rtn
    
    def start(projectname,tag):
        rtn = {'success':False}
        
        if ins_TrainProxy.Run():
            rtn['error'] = "System has in train process"
            return rtn
        
        if not ins_TrainProxy.startTrain(projectname,tag):
            rtn['error'] = "The coach can not start train process"
            return rtn
        
        rtn['success'] = True
        return rtn
    
    
    """
    instance here
    """
    
    def __init__(self):
        self._loop_thread = None
        pass
    
    def Run(self):
        if not self._loop_thread:
            return False
        else:
            return self._loop_thread._running
    
    
    def startTrain(self,projectname,tag):
        if not CoachProxy.init(projectname,tag):
            return False
        self._loop_thread = LoopThread(CoachProxy.doTrain)
        self._loop_thread.start()
        
        return True
    
    def stopTrain(self):
        if self.Run():
            CoachProxy.stop()
            self._loop_thread.stop()
        pass
    
    pass

ins_TrainProxy = TrainProxy()