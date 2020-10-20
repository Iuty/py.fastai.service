
import time
import sys
import os
from IutyLib.file.log import SimpleLog
from IutyLib.commonutil.config import Config
from flask import Flask
from flask_restful import *#Api,Resource
from flask_cors import *
import multiprocessing
from prx.PathProxy import PathProxy

import tensorflow as tf

from api.project_ import *
from api.class_ import *
from api.train_ import *
from api.test_ import *

app_path = PathProxy.app_path
project_path = PathProxy.project_path
PathProxy.mkdir(project_path)

app_log = SimpleLog(os.path.join(app_path,"logs")+"\\")

tf.logging.set_verbosity(tf.logging.ERROR)

config = Config(PathProxy.getConfigPath())

app = Flask(__name__)
api = Api(app)
CORS(app,supports_credentials=True)

api.add_resource(ProjectApi,'/api/project')
api.add_resource(ClassApi,'/api/class')

api.add_resource(TrainApi,'/api/train')
api.add_resource(TestApi,'/api/test')

host = '0.0.0.0'
port = int(config.get("Server","port"))



if __name__ == '__main__':
    multiprocessing.freeze_support()
    app.run(host=host,port=port,debug=False ,use_reloader=False)
    