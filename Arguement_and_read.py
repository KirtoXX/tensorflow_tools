import tensorflow as tf
import numpy as np

'''
1.定义数据提升的方法
2.定义图片裁剪方式
3.定义读取队列
image_tfrecord 内容：lable,height,weight,channel,img_raw 格式:int64,int64,int64,int64,byte
上面读取信息的格式必须和tfrecord制作的格式相同
'''

def restore_data(queue,tfrecord_path = None):
    #files = tf.train.match_filenames_once(tfrecord_path)
    #file_queue = tf.train.string_input_producer(files)
    file_queue = queue
    #定义解析reader
    reader = tf.TFRecordReader()

    #解析tfrecord 协议内存块example
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(serialized_example,
        features={
        "label": tf.FixedLenFeature([], tf.int64),
        "height": tf.FixedLenFeature([], tf.int64),
        "weight": tf.FixedLenFeature([], tf.int64),
        "channel": tf.FixedLenFeature([], tf.int64),
        "img_raw": tf.FixedLenFeature([], tf.string)
        })

    #获取内容 此处返回的是tensor
    label = tf.cast(features["label"],dtype=tf.int32)
    height = tf.cast(features["height"],dtype=tf.int32)
    width = tf.cast(features["weight"],dtype=tf.int32)
    channel = tf.cast(features["channel"],dtype=tf.int32)
    decode_image = tf.decode_raw(features['img_raw'],tf.float32)
    shape = tf.stack([height,width,3])
    decode_image = tf.reshape(decode_image,shape)
    return decode_image,label


def crop_and_resize(image,bbox=None,height=224,width=224):
    #if bbox is None:
    #    bbox = tf.constant([0.0,0.1,1.0,1.0],dtype=tf.float32,shape=[1,1,4])
    #bbox_begin,bbox_size,_ = tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox)
    #image = tf.slice(image,bbox_begin,bbox_size)
    #distorted_image = tf.image.resize_images(distorted_image,size=[height,width])
    #resized_image = tf.reshape(image,shape=[height,width,3])
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    return image

def Arguement(image):
    #在此处设置Arguement 方法
    image = tf.image.random_brightness(image,max_delta=32/255.)
    image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
    image = tf.image.random_hue(image,max_delta=0.2)

    return tf.clip_by_value(image,0.0,0.1)
