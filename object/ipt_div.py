from IutyLib.commonutil.config import Config 
import os
import numpy as np
import tensorflow as tf
from PIL import Image

def checkGroup(config,dirpath,filehead):
    gs = config.Groups()
    formatter = config.Formatter()
    for k in gs:
        fpath = os.path.join(dirpath,filehead+"_"+gs[k]+"."+formatter)
        if not os.path.exists(fpath):
            return False
    return True

def getGroupHead(filename):
    success = False
    fnames = filename.split('_')
    if len(fnames) < 2:
        return success,None
        
    else:
        fgname = ""
        for i in range(0,len(fnames)-1):
            if len(fgname) > 0:
                fgname += "_"
            fgname += fnames[i]
        
        success = True
        return success,fgname
    
    
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
                            #group
                            if config.GroupEnable():
                                succ,fghead = getGroupHead(fsplit[0])
                                
                                if succ:
                                    
                                    if os.path.join(mdir,fghead) in data:
                                        
                                        continue
                                    else:
                                        suc = checkGroup(config,mdir,fghead)
                                        
                                        if suc:
                                            data.append(os.path.join(mdir,fghead))
                                            label.append(int(cls))
                                        else:
                                            continue
                                else:
                                    continue
                            #single
                            else:
                                data.append(os.path.join(mdir,f))
                                label.append(int(cls))
                                
    
    temp = np.array([data, label])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    return image_list,label_list

def readImg(config,files):
    
    image_contents = tf.read_file(files)
    imgd = config.Depth()
    imgf = config.Formatter()
    if imgf == "jpg":
        image = tf.image.decode_jpeg(image_contents, channels=imgd)
    if imgf == "png":
        image = tf.image.decode_png(image_contents,channels=imgd)
    
    return image

def readGroup(config,files):
    #image_contents = tf.read_file(files)
    
    imgw = config.Width()
    imgh = config.Height()
    imgf = config.Formatter()
    groups = config.Groups()
    img_group = []
    
    for k in groups:
        apd = tf.constant("_"+groups[k]+"."+imgf,dtype=tf.string)
        #content = tf.constant(files,shape=files.shape,dtype=tf.string)
        img_content = files + apd
        image_contents = tf.read_file(img_content)
        if imgf == "bmp":
            image=tf.image.decode_bmp(image_contents)
        img_group.append(image)
    image_group = tf.reshape(img_group,[imgw,imgh,len(groups)])
    
    return image_group

def getBatch(config,images,labels):
    imgw = config.Width()
    imgh = config.Height()
    imgd = config.Depth()
    imgf = config.Formatter()
    
    group = config.GroupEnable()
    
    batch_size = config.Batch()
    
    image = tf.cast(images, tf.string)
    label = tf.cast(labels, tf.int32)
    
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    """
    if group:
        image_contents = readFileGroup(input_queue[0])
    else:
        image_contents = tf.read_file(input_queue[0])  # read img from a queue
    """
    # step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    
    if group:
        image = readGroup(config,input_queue[0])
    else:
        image = readImg(config,input_queue[0])
    
    #image = tf.cast(image,dtype=tf.uint8)
    """
    print("*"*30)
    
    print(input_queue[0])
    if imgf == "jpg":
        image = tf.image.decode_jpeg(image_contents, channels=imgd)
    if imgf == "png":
        image = tf.image.decode_png(image_contents,channels=imgd)
    """
    # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image = tf.image.resize_image_with_crop_or_pad(image, imgw, imgh)
    image = tf.image.per_image_standardization(image)
    
    
    # step4：生成batch
    # image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32
    # label_batch: 1D tensor [batch_size], dtype=tf.int32
    
    image_batch, label_batch = tf.train.batch([image, label],batch_size=batch_size,num_threads=4,capacity=max(batch_size,len(images)))
    label_batch = tf.reshape(label_batch, [batch_size])
    
    image_batch = tf.cast(image_batch, tf.float32)
        
    return image_batch,label_batch