import os,time

from PIL import Image
import numpy as np
import tensorflow as tf

from IutyLib.mutithread.threads import LoopThread
from IutyLib.commonutil.config import Config

from prx.PathProxy import PathProxy
from prx.ClassProxy import ClassProxy
from prx.CoachProxy import CoachProxy
#from prx.CoachProxy import ins_CoachProxy
from object.Config import CNNDivParam

import importlib

class TestProxy:
    """
    api here
    """
    def testPicture(project,tag,path):
        rtn = {'success':False}
        
        #if ins_CoachProxy.Run():
            #rtn['error'] = "A train process has run yet"
            #return rtn
        
        
        succ,data = TestProxy.doTest(project,tag,path)
        if not succ:
            rtn['error'] = data
            return rtn
        rtn['data'] = data
        rtn['success'] = True
        return rtn
    
    """
    methods here
    """
    ####abort
    def readPicture(param,path):
        success = False
        #exists?
        img = Image.open(path)
        p_width = param.Width()
        p_height = param.Height()
        
        imag = img.resize([p_width, p_height])
        image = np.array(imag)
        
        success = True
        return success,image
        
    
    def readImage(param,path):
        success = False
        if not os.path.exists(path):
            error = "this image is not exists"
            return success,error
        
        module_ipt = importlib.import_module("object.ipt_div")
        image_array = module_ipt.readImg(param,path)
        
        success = True
        return success,image_array
    
    def readGroup(param,path):
        success = False
        module_ipt = importlib.import_module("object.ipt_div")
        filename = os.path.basename(path)
        filepath = os.path.dirname(path)
        
        succ,headname = module_ipt.getGroupHead(filename)
        if not succ:
            error = "get group head error"
            return success,error
        
        succ = module_ipt.checkGroup(param,filepath,headname)
        if not succ:
            error = "this group is not satisfied group rules"
            return success,error
        grouppath = os.path.join(filepath,headname)
        group_array = module_ipt.readGroup(param,grouppath)
        
        success = True
        return success,group_array
    
    def testOnePicture(projectname,tag,param,image_array):
        success = False
        classes = param.Classes()
        picw = param.Width()
        pich = param.Height()
        picd = param.Depth()
        
        group = param.GroupEnable()
        if group:
            picd = len(param.Groups())
        N_CLASSES = len(classes)
        
        model_type = param.Type()
        if N_CLASSES < 1:
            error = "model has less than one classes"
            return success,error
            
        #with tf.Graph().as_default():
        if True:
            BATCH_SIZE = 1
            
            image = tf.cast(image_array, tf.float32)
            image = tf.image.per_image_standardization(image)
            image = tf.reshape(image, [1, picw, pich, picd])

            logit = CoachProxy.getLogit(model_type,param,image,BATCH_SIZE)

            logit = tf.nn.softmax(logit)

            x = tf.placeholder(tf.float32, shape=[picw, pich, picd])

            # you need to change the directories to yours.
            modelpath = PathProxy.getModelTagDir(projectname,tag)
            savedir = os.path.join(modelpath,"saver")
            

            saver = tf.train.Saver()
        
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                
                ckpt = tf.train.get_checkpoint_state(savedir)
                
                if ckpt and ckpt.model_checkpoint_path:
                    model_name = ckpt.model_checkpoint_path.split('\\')[-1]
                    global_step = model_name.split('-')[-1]
                    saverpath = os.path.join(savedir,model_name)
                    saver.restore(sess, saverpath)
                    print('Loading success, global_step is %s' % global_step)
                    
                else:
                    print(ckpt)
                    print('No checkpoint file found')
                
                sess.graph.as_default()
                img = sess.run(image_array)
                prediction = sess.run(logit, feed_dict={x: img})
                max_index = np.argmax(prediction)
                
                result_data = {"mightbe":None,"mightpercent":None,"result":{}}
                result_data["mightbe"] = classes[str(max_index)]
                result_data["mightpercent"] = str(round(prediction[0, max_index]*100,2))
                for cls in classes:
                    result_data["result"][classes[cls]] = str(round(prediction[0, int(cls)]*100,2))
                
                
                success = True
                
                return success,result_data
    
    
    
    def doTest(projectname,tag,path):
        success = False
        
        param = CNNDivParam(projectname,tag)
        if not param.Exists():
            error = "Model is not exists"
            return success,error
        
        group = param.GroupEnable()
        if group:
            succ,pic_array = TestProxy.readGroup(param,path)
        else:
            succ,pic_array = TestProxy.readImage(param,path)
        if not succ:
            error = pic_array
            return success,error
        #s = datetime.datetime.now()
        success,test_result = TestProxy.testOnePicture(projectname,tag,param,pic_array)
        #e = datetime.datetime.now()
        
    
        return success,test_result
        
