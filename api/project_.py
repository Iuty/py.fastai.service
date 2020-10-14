from flask_restful import Resource
from flask import request
from prx.ProjectProxy import ProjectProxy

class ProjectApi(Resource):
    def initProject():
        _projectname = request.form.get('projectname')
        if not _projectname:
            _projectname = "TestProject"
        setting = request.form
        rtn = ProjectProxy.initProject(_projectname,setting)
        return rtn
    
    def getTagNames():
        rtn = {'success':False}
        _projectname = request.form.get('projectname')
        if not _projectname:
            rtn['error'] = "Projectname is nesserary"
            return rtn
        rtn = ProjectProxy.getTagNames(_projectname)
        return rtn
    
    def post(self):
        _cmd = request.form.get('cmd')
        
        if _cmd == "initProject":
            return ProjectApi.initProject()
        if _cmd == "getProjectNames":
            return ProjectProxy.getProjectNames()
        if _cmd == "getTagNames":
            return ProjectApi.getTagNames()