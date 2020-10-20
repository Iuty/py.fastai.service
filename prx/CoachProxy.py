from object.Config import CNNDivParam,CNNDivSetting

import datetime,os
from prx.PathProxy import PathProxy
from prx.ClassProxy import ClassProxy


import numpy as np
import tensorflow as tf

import importlib

from multiprocessing import Process

class CoachProxy:
    """
    时间戳做模型目录
    """
    def getTimeStamp():
        now = datetime.datetime.now()
        return datetime.datetime.strftime(now,"%Y%m%d_%H%M%S")
    
    
    def createDir(projectname,tag):
        if not tag:
            tag = CoachProxy.getTimeStamp()
        dir = PathProxy.getModelDir(projectname) + tag + "/"
        PathProxy.mkdir(dir)
        return dir,tag
    
    def init(projectname,tag):
        modelpath,tag = CoachProxy.createDir(projectname,tag)
        rtn = ins_CoachProxy.initTrain(projectname,tag)
        
        #CoachProxy.runTensorBoard(os.path.join(modelpath,"log"))
        p = Process(target=CoachProxy.runTensorBoard,args=[os.path.join(modelpath,"log"),])
        p.start()
        
        return rtn
        
        
    def stop():
        ins_CoachProxy.stopTrain()
        CoachProxy.killTensorBoard()
        pass
    
    def runTensorBoard(logdir):
        
        os.system(r"tensorboard --logdir={}".format(logdir))
        
        pass
    
    def killTensorBoard():
        try:
            os.system('taskkill /F /IM tensorboard.exe')
        except Exception as err:
            print(err)
        pass
    
    def doTrain():
        
        while True:
            if ins_CoachProxy.curstep > ins_CoachProxy.period * ins_CoachProxy.maxperiod:
                return False
            
            _,tra_loss,tra_acc = ins_CoachProxy.trainModel()
            #_ = ins_CoachProxy.sess.run([ins_CoachProxy.train_op])
            ins_CoachProxy.curstep += 1
            
            if ins_CoachProxy.curstep % ins_CoachProxy.period == 0:
                
                t_,t_loss,t_acc = ins_CoachProxy.testModel()
                
                scalar_train_loss = tf.summary.scalar('test'+'/loss', tra_loss)
                scalar_train_acc = tf.summary.scalar('test'+'/accuracy', tra_acc)
                
                if t_:
                    scalar_test_loss = tf.summary.scalar('test'+'/loss', t_loss)
                    scalar_test_acc = tf.summary.scalar('test'+'/accuracy', t_acc)
                    summary_op = tf.summary.merge([scalar_train_loss,scalar_train_acc,scalar_test_loss,scalar_test_acc])
                else:
                    summary_op = tf.summary.merge([scalar_train_loss,scalar_train_acc])
                
                ins_CoachProxy.summaryTrain()
                if ins_CoachProxy.curstep // ins_CoachProxy.period % ins_CoachProxy.saveperiod == 0:
                    ins_CoachProxy.saveModel()
                
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (ins_CoachProxy.curstep, tra_loss, tra_acc * 100.0))
                
                if not ins_CoachProxy.runflag:
                    ins_CoachProxy.closeSession()
                
                return None
            pass
    
    def __init__(self):
        self.train_logit = None
        self.train_loss = None
        self.train_op = None
        self.train_acc = None
        
        self.test_logit = None
        self.test_loss = None
        self.test_acc = None
        
        self.runflag = False
        self.curstep = 0
        self.period = 0
        self.maxperiod = 0
        pass
    """
    启动前记录训练参数
    """
    def recordClasses(modeltype,paramcfg,projectname):
        if modeltype == "cnn-div":
            
            classes = ClassProxy.getClasses(projectname)
            for k in classes:
                
                paramcfg.set("Classes",k,classes[k])
        
        pass
    
    def getInput(modeltype,paramcfg,dir):
        if modeltype == "cnn-div":
            module_ipt = importlib.import_module("object.ipt_div")
            return module_ipt.getInputData(paramcfg,dir)
        pass
    
    def getBatch(modeltype,paramcfg,images,labels):
        if modeltype == "cnn-div":
            module_ipt = importlib.import_module("object.ipt_div")
            return module_ipt.getBatch(paramcfg,images,labels)
        pass
    
    def getLogit(modeltype,paramcfg,image_batch,batch_size = None):
        if modeltype == "cnn-div":
            module_mdl = importlib.import_module("object.mdl_div")
            return module_mdl.getLogit(paramcfg,image_batch,batch_size)
        pass
    
    def getLoss(modeltype,paramcfg,logits,labels):
        if modeltype == "cnn-div":
            module_mdl = importlib.import_module("object.mdl_div")
            return module_mdl.getLoss(paramcfg,logits,labels)
        pass
    
    def getTrain(modeltype,paramcfg,loss):
        if modeltype == "cnn-div":
            module_mdl = importlib.import_module("object.mdl_div")
            return module_mdl.getTrain(paramcfg,loss)
        pass
    
    def getEnvaluation(modeltype,paramcfg,logit,labels):
        if modeltype == "cnn-div":
            module_mdl = importlib.import_module("object.mdl_div")
            return module_mdl.getEnvaluation(paramcfg,logit,labels)
        pass
    
    def closeSession(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()
    
    def stopTrain(self):
        self.runflag = False
        pass
    
    def initTrain(self,projectname,tag):
        setting = CNNDivSetting(projectname)
        param = CNNDivParam(projectname,tag)
        
        setting.copy2(param)
        
        modeltype = param.Type()
        modelpath = PathProxy.getModelTagDir(projectname,tag)
        if modeltype == "cnn-div":
            
            CoachProxy.recordClasses(modeltype,param,projectname)
            train_dir = PathProxy.getProjectTrainDir(projectname)
            train_images,train_labels = CoachProxy.getInput(modeltype,param,train_dir)
            
            train_images_batch,train_labels_batch = CoachProxy.getBatch(modeltype,param,train_images,train_labels)
            
            x = tf.placeholder(shape = train_images_batch.shape,dtype=train_images_batch.dtype,name="data_batch")
            y = tf.placeholder(shape = train_labels_batch.shape,dtype=train_labels_batch.dtype,name="label_batch")
            
            train_logit = CoachProxy.getLogit(modeltype,param,x)
            
            train_loss = CoachProxy.getLoss(modeltype,param,train_logit,y)
            
            train_op = CoachProxy.getTrain(modeltype,param,train_loss)
            
            train_acc = CoachProxy.getEnvaluation(modeltype,param,train_logit,y)
            self.train_images_batch = train_images_batch
            self.train_labels_batch = train_labels_batch
            self.train_logit = train_logit
            self.train_loss = train_loss
            self.train_op = train_op
            self.train_acc = train_acc
            
        
            
            test_dir = PathProxy.getProjectTestDir(projectname)
            test_images,test_labels = CoachProxy.getInput(modeltype,param,test_dir)
            
            self.test_dir = test_dir
            self.test_images = test_images
        
            if len(test_images) > 0:
                test_images_batch,test_labels_batch = CoachProxy.getBatch(modeltype,param,test_images,test_labels)
            
                self.test_images_batch = test_images_batch
                self.test_labels_batch = test_labels_batch
                
        
        
        #self.summary_op = tf.summary.merge([scalar_tra_loss,scalar_tra_acc,scalar_test_loss,scalar_test_acc])
        self.dict_summary = {"train/loss":0.0,"train/accuracy":0.0,"test/loss":0.0,"test/accuracy":0.0}
        list_merge = []
        for k in self.dict_summary:
            v = self.getSummary(k)
            sc = tf.summary.scalar(k, v)
            list_merge.append(sc)
        self.summary_op = tf.summary.merge(list_merge)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        
        self.period = param.Period()
        self.saveperiod = param.SavePeriod()
        self.maxperiod = param.MaxPeriod()
        self.curstep = 0
        
        self.runflag = True
        
        self.savedir = os.path.join(modelpath,"saver")
        PathProxy.mkdir(self.savedir)
        
        self.logdir = os.path.join(modelpath,"log")
        PathProxy.mkdir(self.logdir)
        
        self.writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        #print(CoachProxy.getInput(modeltype,param,train_dir))
        
        return True
    
    def saveModel(self):
        checkpoint_path = os.path.join(self.savedir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=self.curstep)
        pass
    
    def trainModel(self):
        graph = self.sess.graph
        
        x = graph.get_tensor_by_name("data_batch:0")
        y = graph.get_tensor_by_name("label_batch:0")
        x_,y_ = self.sess.run([self.train_images_batch,self.train_labels_batch])
        _, tra_loss, tra_acc = self.sess.run([self.train_op, self.train_loss, self.train_acc],feed_dict={x:x_,y:y_})
        self.dict_summary["train/loss"] = tra_loss
        self.dict_summary["train/accuracy"] = tra_acc
        return _,tra_loss,tra_acc
    
    def testModel(self):
        if len(self.test_images) == 0:
            return False,None,None
        graph = self.sess.graph
        
        x = graph.get_tensor_by_name("data_batch:0")
        y = graph.get_tensor_by_name("label_batch:0")
        x_,y_ = self.sess.run([self.test_images_batch,self.test_labels_batch])
        t_logit,t_loss,t_acc = self.sess.run([self.train_logit,self.train_loss,self.train_acc],feed_dict={x:x_,y:y_})
        self.dict_summary["test/loss"] = t_loss
        self.dict_summary["test/accuracy"] = t_acc
        return True,t_loss,t_acc
        pass
    
    def getSummary(self,key):
        if key in self.dict_summary:
            return tf.constant(self.dict_summary[key],dtype=tf.float32)
        else:
            return tf.constant(0.0,dtype=tf.float32)
    
    def summaryTrain(self):
        print(self.dict_summary)
        summary_op = self.sess.run(self.summary_op)
        self.writer.add_summary(summary_op,self.curstep)
        pass
    
    pass
    
ins_CoachProxy = CoachProxy()