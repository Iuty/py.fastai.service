from flask_restful import Resource
from flask import request
from prx.ClassProxy import ClassProxy

class ClassApi(Resource):
    def addClass():
        _projectname = request.form.get('projectname')
        if not _projectname:
            _projectname = "TestProject"
        
        _classname = request.form.get('classname')
        if not _classname:
            _classname = "TestClass"
        
        rtn = ClassProxy.addClass(_projectname,_classname)
        return rtn
    
    def getClassNames():
        rtn = {'success':False}
        _projectname = request.form.get('projectname')
        if not _projectname:
            rtn['error'] = "projectname is nesserary"
            return rtn
        return ClassProxy.getClassNames(_projectname)
    
    def post(self):
        _cmd = request.form.get('cmd')
        
        if _cmd == "getClassNames":
            return ClassApi.getClassNames()
        
        if _cmd == "addClass":
            return ClassApi.addClass()