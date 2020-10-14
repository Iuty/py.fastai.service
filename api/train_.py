from flask_restful import Resource
from flask import request
from prx.TrainProxy import TrainProxy

class TrainApi(Resource):
    def start():
        _projectname = request.form.get('projectname')
        _tag = request.form.get('tag')
        
        
        return TrainProxy.start(_projectname,_tag)
    
    def stop():
        return TrainProxy.stop()
    
    def post(self):
        _cmd = request.form.get('cmd')
        
        if _cmd == "start":
            return TrainApi.start()
        if _cmd == "stop":
            return TrainApi.stop()