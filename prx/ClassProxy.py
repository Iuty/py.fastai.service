from IutyLib.commonutil.config import Config
from prx.PathProxy import PathProxy
from prx.ProjectProxy import ProjectProxy
import os

class ClassProxy:
    """
    methods here
    """
    def getClasses(projectname):
        train_path = PathProxy.getProjectTrainDir(projectname)
        
        for maindir,pdir,etcfile in os.walk(train_path):
            if maindir == train_path:
                rtn = {}
                for p in pdir:
                    """
                    ps = p.split('_')
                    
                    if len(ps) == 2:
                        try:
                            mark = int(ps[0])
                            
                            rtn[ps[0]] = ps[1]
                        except Exception as error:
                            a = 0
                    """
                    rtn[str(len(rtn))] = p
                return rtn
        return {}
    
    def getTagClasses(projectname,tag):
        default = {}
        tagdir = PathProxy.getModelParamPath(projectname,tag)
        if not os.path.exists(tagdir):
            return default
        
        cfg = Config(tagdir)
        for i in range(0,1000):
            classname = cfg.get("Classes",str(i))
            if classname:
                default[str(i)] = classname
            else:
                break
        
        return default
    
    def addClassDir(projectname,dirname):
        train_dir = PathProxy.getProjectTrainDir(projectname)
        PathProxy.mkdir(train_dir+dirname)
        
        test_dir = PathProxy.getProjectTestDir(projectname)
        PathProxy.mkdir(test_dir+dirname)
        pass
    """
    api here
    """
    def getClassNames(projectname):
        rtn = {'success':False}
        if not ProjectProxy.isExists(projectname):
            rtn['error'] = "can not get class name because it not exists"
            return rtn
        
        rtn['success'] = True
        rtn['data'] = list(ClassProxy.getClasses(projectname).values())
        return rtn
        
    def addClass(projectname,classname):
        rtn = {'success':False}
        classnames = ClassProxy.getClassNames(projectname)
        if not classnames['success']:
            rtn['error'] = classnames['error']
            return rtn
        
        if classname in classnames['data']:
            rtn['error'] = classname + " has exists in "+ projectname
            return rtn
        
        #dirname = str(len(classnames['data'])) + '_' + classname
        dirname = classname
        ClassProxy.addClassDir(projectname,dirname)
        rtn['success'] = True
        return rtn
    pass