from IutyLib.commonutil.config import Config
from prx.PathProxy import PathProxy
from object.Config import CNNDivSetting
import os

class ProjectProxy:
    
    def setSetting(cfg,setting,section,key,default):
        v = setting.get(key)
        if not v:
            v = default
        cfg.set(section,key,v)
        pass
    
    
    def isExists(projectname):
        proj_path = PathProxy.getProjectDir(projectname)
        return os.path.exists(proj_path)
    
    def initProject(projectname,setting):
        rtn = {'success':False}
        
        proj_path = PathProxy.getProjectDir(projectname)
        if os.path.exists(proj_path):
            rtn['error'] = "can not init project because it has exists"
            return rtn
        
        PathProxy.mkdir(proj_path)
        PathProxy.mkdir(os.path.join(proj_path,"train"))
        PathProxy.mkdir(os.path.join(proj_path,"model"))
        PathProxy.mkdir(os.path.join(proj_path,"test"))
        # cnn type here
        
        psetting = CNNDivSetting(projectname)
        psetting.createConfig()
        
        rtn['success'] = True
        return rtn
        
    def getProjectNames():
        rtn = {'success':False}
        for maindir,pdir,etcfile in os.walk(PathProxy.project_path):
            if maindir == PathProxy.project_path:
                rtn['success'] = True
                rtn['data'] = pdir
                return rtn
                
        rtn['error'] = "can not find projects path"
        return rtn
        pass
    pass
    
    def getTagNames(projectname):
        rtn = {'success':False}
        if not ProjectProxy.isExists(projectname):
            rtn['error'] = "This project is not exists"
            return rtn
        modelpath = PathProxy.getModelDir(projectname)
        for maindir,pdir,etcfile in os.walk(modelpath):
            if maindir == modelpath:
                rtn['success'] = True
                rtn['data'] = pdir
                return rtn
        rtn['error'] = "some unknown error"
        return rtn