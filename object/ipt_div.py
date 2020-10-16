from IutyLib.commonutil.config import Config 
import os
import numpy as np
import tensorflow as tf



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