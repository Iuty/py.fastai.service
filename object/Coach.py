from IutyLib.commonutil.config import Config 
import os
import numpy as np
import tensorflow as tf

class CNNDivCoach:
    
    def getInputData(config,dir):
        formatter = config.Formatter()
        data = []
        label = []
        classes = config.Classes()
        for cls in classes:
            #for mdir,subdir,filename in os.walk(dir+cls+"_"+classes[cls]):
            for mdir,subdir,filename in os.walk(dir+classes[cls]):
                if len(filename) > 0:
                    for f in filename:
                        fsplit = f.split('.')
                        if len(fsplit)>1:
                            if fsplit[-1] == formatter:
                                data.append(os.path.join(mdir,f))
                                label.append(int(cls))
                                
        
        temp = np.array([data, label])
        temp = temp.transpose()
        np.random.shuffle(temp)
        image_list = list(temp[:, 0])
        label_list = list(temp[:, 1])
        label_list = [int(float(i)) for i in label_list]
        return image_list,label_list
        
    
    def getBatch(config,images,labels):
        imgw = config.Width()
        imgh = config.Height()
        imgd = config.Depth()
        imgf = config.Formatter()
        
        batch_size = config.Batch()
        
        image = tf.cast(images, tf.string)
        label = tf.cast(labels, tf.int32)
        
        input_queue = tf.train.slice_input_producer([image, label])
        label = input_queue[1]
        image_contents = tf.read_file(input_queue[0])  # read img from a queue
        
        # step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
        if imgf == "jpg":
            image = tf.image.decode_jpeg(image_contents, channels=imgd)
        
        # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
        image = tf.image.resize_image_with_crop_or_pad(image, imgw, imgh)
        image = tf.image.per_image_standardization(image)
        
        # step4：生成batch
        # image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32
        # label_batch: 1D tensor [batch_size], dtype=tf.int32
        image_batch, label_batch = tf.train.batch([image, label],batch_size=batch_size,num_threads=32,capacity=max(batch_size,len(images)))
        label_batch = tf.reshape(label_batch, [batch_size])
        
        image_batch = tf.cast(image_batch, tf.float32)
        
        return image_batch,label_batch
    
    def getLogit(config,image_batch,batch_size):
        if batch_size == None:
            batch_size = config.Batch()
        conv1_depth = config.Conv1Depth()
        conv2_depth = config.Conv2Depth()
        class_count = len(config.Classes())
        with tf.variable_scope('conv1') as scope:
            weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, conv1_depth], stddev=1.0, dtype=tf.float32),
                                name='weights', dtype=tf.float32)

            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[conv1_depth]),
                             name='biases', dtype=tf.float32)

            conv = tf.nn.conv2d(image_batch, weights, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)

        # 池化层1
        # 3x3最大池化，步长strides为2，池化后执行lrn()操作，局部响应归一化，对训练有利。
        with tf.variable_scope('pooling1_lrn') as scope:
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
            norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        # 卷积层2
        # 16个3x3的卷积核（16通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
        with tf.variable_scope('conv2') as scope:
            weights = tf.Variable(tf.truncated_normal(shape=[3, 3, conv1_depth, conv2_depth], stddev=0.1, dtype=tf.float32),
                                name='weights', dtype=tf.float32)

            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[conv2_depth]),
                                name='biases', dtype=tf.float32)

            conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name='conv2')

        # 池化层2
        # 3x3最大池化，步长strides为2，池化后执行lrn()操作，
        # pool2 and norm2
        with tf.variable_scope('pooling2_lrn') as scope:
            norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling2')

        # 全连接层3
        # 128个神经元，将之前pool层的输出reshape成一行，激活函数relu()
        with tf.variable_scope('local3') as scope:
            reshape = tf.reshape(pool2, shape=[batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = tf.Variable(tf.truncated_normal(shape=[dim, class_count*4], stddev=0.005, dtype=tf.float32),
                                name='weights', dtype=tf.float32)

            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[class_count*4]),
                                name='biases', dtype=tf.float32)
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        # 全连接层4
        # 128个神经元，激活函数relu()
        with tf.variable_scope('local4') as scope:
            weights = tf.Variable(tf.truncated_normal(shape=[class_count*4, class_count*4], stddev=0.005, dtype=tf.float32),
                                name='weights', dtype=tf.float32)

            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[class_count*4]),
                                name='biases', dtype=tf.float32)

            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

        # dropout层
        #    with tf.variable_scope('dropout') as scope:
        #        drop_out = tf.nn.dropout(local4, 0.8)

        # Softmax回归层
        # 将前面的FC层输出，做一个线性回归，计算出每一类的得分
        with tf.variable_scope('softmax_linear') as scope:
            weights = tf.Variable(tf.truncated_normal(shape=[class_count*4, class_count], stddev=0.005, dtype=tf.float32),
                                name='softmax_linear', dtype=tf.float32)

            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[class_count]),
                             name='biases', dtype=tf.float32)

            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

        return softmax_linear
        
    def getLoss(config,logits,labels):
        
        with tf.variable_scope('loss') as scope:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                        name='xentropy_per_example')
            loss = tf.reduce_mean(cross_entropy, name='loss')
            tf.summary.scalar(scope.name + '/loss', loss)
        return loss
    
    def getTrain(config,loss):
        learning_rate = config.LearnRate()
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
    
    def getEnvaluation(config,logits,labels):
        with tf.variable_scope('accuracy') as scope:
            correct = tf.nn.in_top_k(logits, labels, 1)
            correct = tf.cast(correct, tf.float16)
            accuracy = tf.reduce_mean(correct)
            tf.summary.scalar(scope.name + '/accuracy', accuracy)
        return accuracy
    
    def getTrainMethods(projectname,modelpath,cfg):
        
        envalueation = None
        
        param = CNNDivCoach.recordStartParam(projectname,modelpath,cfg)
        
        
        classes = ClassProxy.getClasses(projectname)
        
        train_dir = PathProxy.getProjectTrainDir(projectname)
        
        train_data,train_label = CNNDivCoach.getInputData(classes,train_dir,param)
        
        test_dir = PathProxy.getProjectTestDir(projectname)
        test_data,test_label = CNNDivCoach.getInputData(classes,test_dir,param)
        
        
        imgw = int(cfg.get("Image","width"))
        imgh = int(cfg.get("Image","height"))
        imgd = int(cfg.get("Image","depth"))
        batch_size = int(cfg.get("Train","batch"))
        
        train_image_batch,train_label_batch = CNNDivCoach.getBatch(train_data,train_label,param)
        test_image_batch,test_label_batch = CNNDivCoach.getBatch(test_data,test_label,param)
        
        
        conv1_depth = int(cfg.get("Logit","conv1depth"))
        conv2_depth = int(cfg.get("Logit","conv2depth"))
        class_count = len(classes)
        
        train_logit = CNNDivCoach.getLogit(train_image_batch,batch_size,conv1_depth,conv2_depth,class_count)
        test_logit = CNNDivCoach.getLogit(test_image_batch,batch_size,conv1_depth,conv2_depth,class_count)
        
        
        train_loss = CNNDivCoach.getLoss(train_logit,train_label_batch)
        test_loss = CNNDivCoach.getLoss(test_logit,test_label_batch)
        
        
        learning_rate = float(cfg.get("Train","learnrate"))
        train_op = CNNDivCoach.getTrain(train_loss,learning_rate)
        
        
        train_acc = CNNDivCoach.getEnvaluation(train_logit,train_label_batch)
        test_acc = CNNDivCoach.getEnvaluation(test_logit,test_label_batch)
        return train_logit,train_loss,train_op,train_acc,test_logit,test_loss,test_acc
    
    def getCoachMethods(projectname,modelpath,cfg):
        try:
            return True,CNNDivCoach.getTrainMethods(projectname,modelpath,cfg)
        except Exception as err:
            return False,None
    
    
    pass