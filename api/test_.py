from flask_restful import Resource
from flask import request
from prx.TrainProxy import TrainProxy
from prx.TestProxy import TestProxy

class TestApi(Resource):
    def testPicture():
        rtn = {'success':False}
        _projectname = request.form.get('projectname')
        if not _projectname:
            rtn['error'] = "projectname is nesserary"
            return rtn
        
        _tag = request.form.get('tag')
        if not _tag:
            rtn['error'] = "tag is nesserary"
            return rtn
        
        _path = request.form.get('path')
        if not _path:
            rtn['error'] = "path is nesserary"
            return rtn
        
        
        return TestProxy.testPicture(_projectname,_tag,_path)
    
    
    def testDirectory():
        rtn = {'success':False}
        _projectname = request.form.get('projectname')
        if not _projectname:
            rtn['error'] = "projectname is nesserary"
            return rtn
        
        _tag = request.form.get('tag')
        if not _tag:
            rtn['error'] = "tag is nesserary"
            return rtn
        
        _path = request.form.get('path')
        if not _path:
            rtn['error'] = "path is nesserary"
            return rtn
        
        
        return TestProxy.testDirectory(_projectname,_tag,_path)
    
    def post(self):
        _cmd = request.form.get('cmd')
        
        if _cmd == "testPicture":
            return TestApi.testPicture()
        
        if _cmd == "testDirectory":
            return TestApi.testDirectory()
