from flask_restful import Resource
from flask import request
from prx.TrainProxy import TrainProxy,ins_TrainProxy
from prx.TestProxy import TestProxy
from prx.ProjectProxy import ProjectProxy

class FastCnnApi(Resource):
    def start():
        _projectname = request.form.get('projectname')
        _tag = request.form.get('tag')
        
        if not _projectname:
            return {"success":False,"error":"projectname is nesserary"}
        
        return TrainProxy.start(_projectname,_tag)
    
    def stop():
        return TrainProxy.stop()
    
    def testPictures():
    
        path = request.form.get('path')
        tdir = request.form.get('testdir')
        
        if not path:
            return {"success":False,"error":"path is nesserary"}
        
        
    def getProjectNames():
        rtn = ProjectProxy.getProjectNames()
        return rtn
    
    def getProjectTags():
        _projectname = request.form.get('projectname')
        
        if not _projectname:
            return {"success":False,"error":"projectname is nesserary"}
        rtn = ProjectProxy.getTagNames(_projectname)
        return rtn
    
    def post(self):
        _cmd = request.form.get('cmd')
        
        if _cmd == "start":
            rtn = FastCnnApi.start()
            
        if _cmd == "stop":
            rtn = FastCnnApi.stop()
        
        if _cmd == "status":
            rtn = {"success":True}
        
        if _cmd == "testPictures":
            rtn = FastCnnApi.testPictures()
            
        if _cmd == "getProjectNames":
            rtn = FastCnnApi.getProjectNames()
        
        if _cmd == "getProjectTags":
            rtn = FastCnnApi.getProjectTags()
        
        rtn["status"] = ins_TrainProxy.Run()
        
        return rtn