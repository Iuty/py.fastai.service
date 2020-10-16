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
            
            _, tra_loss, tra_acc = ins_CoachProxy.sess.run([ins_CoachProxy.train_op, ins_CoachProxy.train_loss, ins_CoachProxy.train_acc])
            #_ = ins_CoachProxy.sess.run([ins_CoachProxy.train_op])
            ins_CoachProxy.curstep += 1
            
            if ins_CoachProxy.curstep % ins_CoachProxy.period == 0:
                
                if ins_CoachProxy.curstep // ins_CoachProxy.period % ins_CoachProxy.saveperiod == 0:
                    checkpoint_path = os.path.join(ins_CoachProxy.savedir, 'model.ckpt')
                    ins_CoachProxy.saver.save(ins_CoachProxy.sess, checkpoint_path, global_step=ins_CoachProxy.curstep)
                    if ins_CoachProxy.test_logit != None:
                        with tf.Session() as sess:
                            sess.run(tf.global_variables_initializer())
                
                            ckpt = tf.train.get_checkpoint_state(ins_CoachProxy.savedir)
                
                            if ckpt and ckpt.model_checkpoint_path:
                                model_name = ckpt.model_checkpoint_path.split('\\')[-1]
                                global_step = model_name.split('-')[-1]
                                saverpath = os.path.join(ins_CoachProxy.savedir,model_name)
                                saver.restore(sess, saverpath)
                                sess.graph.as_default()
            
                                test_loss, test_acc = sess.run([ins_CoachProxy.test_loss, ins_CoachProxy.test_acc])
                                
                summary_op = ins_CoachProxy.sess.run(ins_CoachProxy.summary_op)
                ins_CoachProxy.writer.add_summary(summary_op,ins_CoachProxy.curstep)
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
            
            train_logit = CoachProxy.getLogit(modeltype,param,train_images_batch)
            
            train_loss = CoachProxy.getLoss(modeltype,param,train_logit,train_labels_batch)
            
            train_op = CoachProxy.getTrain(modeltype,param,train_loss)
            
            train_acc = CoachProxy.getEnvaluation(modeltype,param,train_logit,train_labels_batch)
            
            self.train_logit = train_logit
            self.train_loss = train_loss
            self.train_op = train_op
            self.train_acc = train_acc
            
        
            
            test_dir = PathProxy.getProjectTestDir(projectname)
            test_images,test_labels = CoachProxy.getInput(modeltype,param,test_dir)
            
            if len(test_images) > 0:
                test_images_batch,test_labels_batch = CoachProxy.getBatch(modeltype,param,test_images,test_labels)
            
                test_logit = CoachProxy.getLogit(modeltype,param,test_images_batch)
            
                test_loss = CoachProxy.getLoss(modeltype,param,test_logit,test_labels_batch)
            
                test_acc = CoachProxy.getEnvaluation(modeltype,param,test_logit,test_labels_batch)
                
                self.test_logit = test_logit
                self.test_loss = test_loss
                
                self.test_acc = test_acc
            else:
                self.test_logit = None
                self.test_loss = None
                
                self.test_acc = None
        
        scalar_loss = tf.summary.scalar('train'+'/loss', self.train_loss)
        scalar_acc = tf.summary.scalar('train'+'/accuracy', self.train_acc)
        
        
        if self.test_logit != None:
            scalar_test_loss = tf.summary.scalar('test'+'/loss', self.test_loss)
            scalar_test_acc = tf.summary.scalar('test'+'/accuracy', self.test_acc)
            self.summary_op = tf.summary.merge([scalar_loss,scalar_acc,scalar_test_loss,scalar_test_acc])
            
        else:
            self.summary_op = tf.summary.merge([scalar_loss,scalar_acc])
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
        
        return True
    
    
    
    pass
    
ins_CoachProxy = CoachProxy()