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
        
        
        succ,data = TestProxy.doTestImage(project,tag,path)
        if not succ:
            rtn['error'] = data
            return rtn
        rtn['data'] = data
        rtn['success'] = True
        return rtn
    
    def testDirectory(project,tag,path):
        rtn = {'success':False}
        
        succ,data = TestProxy.doTestDirectory(project,tag,path)
        if not succ:
            rtn['error'] = data
            return rtn
        rtn['data'] = data
        rtn['statistics'] = TestProxy.getStatistics(data)
        rtn['success'] = True
        return rtn
        
    """
    methods here
    """
    def getStatistics(test_result):
        result = {}
        for item in test_result:
            if not item["result"] in result:
                result[item["result"]] = 0
            result[item["result"]] += 1
        return result
    
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
    
    def testOnePicture(projectname,tag,param,image_array,image_name,BATCH_SIZE=1):
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
        
            image = tf.cast(image_array, tf.float32)
            image = tf.image.per_image_standardization(image)
            image = tf.reshape(image, [BATCH_SIZE, picw, pich, picd])

            logit = CoachProxy.getLogit(model_type,param,image,BATCH_SIZE)

            logit = tf.nn.softmax(logit)

            x = tf.placeholder(tf.float32, shape=[BATCH_SIZE,picw, pich, picd])

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
                result_data = []
                for i in range(BATCH_SIZE):
                    mightbe = np.argmax(prediction[i])
                    d = {"image":image_name[i],"result":classes[str(mightbe)],"percent":str(round(prediction[i][mightbe]*100.0,2))}
                    result_data.append(d)
                
                
                success = True
                
                return success,result_data
    
    
    
    def doTestImage(projectname,tag,path):
        success = False
        
        if not os.path.isfile(path):
            error = "path is not a file or path is not exists"
            return success,error
        
        param = CNNDivParam(projectname,tag)
        if not param.Exists():
            error = "Model is not exists"
            return success,error
        
        group = param.GroupEnable()
        imgw = param.Width()
        imgh = param.Height()
        imgd = param.Depth()
        if group:
            imgd = len(param.Groups())
            succ,pic_array = TestProxy.readGroup(param,path)
        else:
            succ,pic_array = TestProxy.readImage(param,path)
        if not succ:
            error = pic_array
            return success,error
        tf_pic_array = tf.reshape(pic_array,shape=[1,imgw,imgh,imgd])
        #s = datetime.datetime.now()
        success,test_result = TestProxy.testOnePicture(projectname,tag,param,tf_pic_array,os.path.basename(path))
        #e = datetime.datetime.now()
        
    
        return success,test_result
    
    def doTestDirectory(projectname,tag,path):
        success = False
        
        if not os.path.isdir(path):
            error = "path is not a directory or path is not exists"
            return success,error
        
        param = CNNDivParam(projectname,tag)
        if not param.Exists():
            error = "Model is not exists"
            return success,error
        
        module_ipt = importlib.import_module("object.ipt_div")
        file_list,label_list = module_ipt.walkDir(param,path)
        
        group = param.GroupEnable()
        imgw = param.Width()
        imgh = param.Height()
        imgd = param.Depth()
        
        if group:
            imgd = len(param.Groups())
        data = []
        pic_array_list = []
        for f in file_list:
            
            if group:
                dv = param.Groups()["0"]
                fmt = param.Formatter()
                fname = f + "_" + dv + "." + fmt
                succ,pic_array = TestProxy.readGroup(param,fname)
            else:
                succ,pic_array = TestProxy.readImage(param,f)
            if not succ:
                continue
            pic_array_list.append(pic_array)
        tf_pic_array = tf.reshape(pic_array_list,shape=[len(pic_array_list),imgw,imgh,imgd])
        
        succ,test_result = TestProxy.testOnePicture(projectname,tag,param,tf_pic_array,[os.path.basename(pt) for pt in file_list],len(pic_array_list))
        print(test_result)
        """
        if not succ:
            continue
        d = {"file":os.path.basename(f),"data":test_result}
        data.append(d)
        #s = datetime.datetime.now()
        print(d)
        os.system("pause")
        #e = datetime.datetime.now()
        """
        #success = True
        return succ,test_result
