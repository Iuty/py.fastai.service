from IutyLib.commonutil.config import Config 
import os
import numpy as np
import tensorflow as tf

    
def getLogit(config,image_batch,batch_size):
    
    if batch_size == None:
        batch_size = config.Batch()
    imgd = config.Depth()
    if config.GroupEnable():
        imgd = len(config.Groups())
        #imgd += (imgd * (imgd-1))//2
    conv1_depth = config.Conv1Depth()
    conv2_depth = config.Conv2Depth()
    class_count = len(config.Classes())
    
    
    with tf.variable_scope('conv1') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[3, 3, imgd, conv1_depth*imgd], stddev=1.0, dtype=tf.float32),
                            name='weights', dtype=tf.float32)
        
        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[conv1_depth*imgd]),
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
        weights = tf.Variable(tf.truncated_normal(shape=[3, 3, conv1_depth*imgd, conv2_depth*imgd], stddev=0.1, dtype=tf.float32),
                            name='weights', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[conv2_depth*imgd]),
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
        weights = tf.Variable(tf.truncated_normal(shape=[dim, imgd*class_count*class_count], stddev=0.005, dtype=tf.float32),
                            name='weights', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[imgd*class_count*class_count]),
                            name='biases', dtype=tf.float32)
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # 全连接层4
    # 128个神经元，激活函数relu()
    with tf.variable_scope('local4') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[imgd*class_count*class_count, imgd*class_count], stddev=0.005, dtype=tf.float32),
                            name='weights', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[imgd*class_count]),
                            name='biases', dtype=tf.float32)

        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # dropout层
    #    with tf.variable_scope('dropout') as scope:
    #        drop_out = tf.nn.dropout(local4, 0.8)

    # Softmax回归层
    # 将前面的FC层输出，做一个线性回归，计算出每一类的得分
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[imgd*class_count, class_count], stddev=0.005, dtype=tf.float32),
                            name='softmax_linear', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[class_count]),
                            name='biases', dtype=tf.float32)

        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
    
    return softmax_linear
        
def getLoss(config,logits,labels):
    
    with tf.variable_scope('loss') as scope:
        #sparse_softmax_cross_entropy_with_logits
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


