import os

class PathProxy:
    """
    path define here
    """
    app_path = r"d:/FastCNN/"
    project_path = app_path + r"Projects/"
    def getConfigPath():
        return PathProxy.app_path + r"Config/Config.conf"
    
    def getProjectDir(projectname):
        return PathProxy.project_path + projectname + "/"
    
    def getSettingPath(projectname):
        return PathProxy.getProjectDir(projectname) + 'Setting.conf'
    
    def getProjectTrainDir(projectname):
        return PathProxy.getProjectDir(projectname) + "train" + "/"
    
    def getProjectTestDir(projectname):
        return PathProxy.getProjectDir(projectname) + "test" + "/"
    
    def getClassDir(projectname,classname):
        return PathProxy.getProjectDir(projectname) + classname + "/"
        
    def getModelDir(projectname):
        return PathProxy.getProjectDir(projectname) + "model/"
    
    def getModelTagDir(projectname,tag):
        return PathProxy.getModelDir(projectname) + tag +"/"
        
    def getModelParamPath(projectname,tag):
        return PathProxy.getModelTagDir(projectname,tag) + 'Param.conf'
    
    """
    method here
    """
    def mkdir(dir):
        if os.path.exists(dir):
            return
        os.makedirs(dir)
        pass