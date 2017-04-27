import tensorflow as tf
import os
from scipy import misc

'''
制作tfrecord  二进制数据
将数据存放在data文件夹下面
image_tfrecord 内容：lable,height,weight,channel,img_raw 格式:int,int,int,int,byte
'''

data_path = 'data/'
tfrecord_path = 'image_tfrecord.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecord_path)

class1 = os.listdir(data_path)   #统计全部class
print('your class:')
print(class1)
for index,name in enumerate(class1):   #python 枚举
    class_path = data_path+name +'/'
    images_path = os.listdir(class_path)

    print('process class '+name+'...')
    for signle_image in images_path:
        temp_path = class_path+signle_image
        #img = Image.open(temp_path)
        #img = img.resize((224,224))
        #img = tf.image.decode_image(tf.read_file(temp_path))
        img = misc.imread(temp_path,'RGB')
        height,weight= img.shape
        channel = 3
        img_raw = img.tostring()  # 将图片转化为原生string

        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'weight': tf.train.Feature(int64_list=tf.train.Int64List(value=[weight])),
            'channel': tf.train.Feature(int64_list=tf.train.Int64List(value=[channel])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))

        writer.write(example.SerializeToString())  # 序列化为字符串
print('done.')
writer.close()
