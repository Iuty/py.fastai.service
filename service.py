# Usage:
    # service.exe install
    # service.exe start
    # service.exe stop
    # service.exe remove

    # you can see output of this program running python site-packages win32libwin32traceutil

import win32service
import win32serviceutil
import win32event
import win32evtlogutil
#import win32traceutil
import servicemanager
import winerror
import time
import sys
import os
from IutyLib.file.log import SimpleLog
from flask import Flask
from flask_restful import *#Api,Resource
from flask_cors import *
from multiprocessing import Process


class cnnservice(win32serviceutil.ServiceFramework):
    _svc_name_ = "FastCNN"                    #服务名
    _svc_display_name_ = "Fast AI"                 #job在windows services上显示的名字
    _svc_description_ = "CNN System"        #job的描述
    
    _svc_path = "d:\\FastCNN"
    _log = SimpleLog(os.path.join(_svc_path,"logs")+"\\")
    
    app = Flask(__name__)
    api = Api(app)
    CORS(app,supports_credentials=True)
    
    host = '0.0.0.0'
    port = 7738
    
    def __init__(self,args):
        
        win32serviceutil.ServiceFramework.__init__(self,args)
        self.hWaitStop=win32event.CreateEvent(None, 0, 0, None)
        process = None
        

    def SvcStop(self):
        
        # tell Service Manager we are trying to stop (required)
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)

        # write a message in the SM (optional)
        # import servicemanager
        # servicemanager.LogInfoMsg("cnnservice - Recieved stop signal")
        
        # set the event to call
        self._log.info('1')
        
        #if self.process:
        self._log.info('2')
        self.process.terminate()
        self._log.info('3')
        self.process.join()
        self._log.info('4')
        
        win32event.SetEvent(self.hWaitStop)
        self._log.info('5')
        win32evtlogutil.ReportEvent(self._svc_name_,servicemanager.PYS_SERVICE_STOPPED, 0,servicemanager.EVENTLOG_INFORMATION_TYPE,(self._svc_name_, ''))
        self.ReportServiceStatus(win32service.SERVICE_STOPPED)

    def SvcDoRun(self):
        
        import servicemanager
        # Write a 'started' event to the event log... (not required)
        #
        #win32evtlogutil.ReportEvent(self._svc_name_,servicemanager.PYS_SERVICE_STARTED, 0,
                                    #servicemanager.EVENTLOG_INFORMATION_TYPE,(self._svc_name_, ''))

        # methode 1: wait for beeing stopped ...
        # win32event.WaitForSingleObject(self.hWaitStop, win32event.INFINITE)

        # methode 2: wait for beeing stopped ...
        self.timeout=1000  # In milliseconds (update every second)

        try:
            #self.process = Process(target=cnnservice.app.run,kwargs={'host':cnnservice.host, 'port':cnnservice.port, 'debug' : True, 'use_reloader':False})
            self.process = Process(target=cnnservice.app.run(host=cnnservice.host, port=cnnservice.port, debug=True, use_reloader=False))
            #self.process.start()
        except Exception as err:
            self._log.error(str(err))
        # and write a 'stopped' event to the event log (not required)
        #
        #win32evtlogutil.ReportEvent(self._svc_name_,servicemanager.PYS_SERVICE_STOPPED, 0,servicemanager.EVENTLOG_INFORMATION_TYPE,(self._svc_name_, ''))

        #self.ReportServiceStatus(win32service.SERVICE_STOPPED)

        return

if __name__ == '__main__':

    # if called without argvs, let's run !
    if len(sys.argv) == 1:
        try:
            evtsrc_dll = os.path.abspath(servicemanager.__file__)
            servicemanager.PrepareToHostSingle(cnnservice)
            servicemanager.Initialize('cnnservice', evtsrc_dll)
            servicemanager.StartServiceCtrlDispatcher()
        except Exception as err:
            a = 1
    else:
        win32serviceutil.HandleCommandLine(cnnservice)
    
        