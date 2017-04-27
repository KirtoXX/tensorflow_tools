import tensorflow as tf
import network
import Arguement_and_read as preprocessing

def train():
    tfrecord_path = 'tfrecord/image_tfrecord.tfrecords'
    nb_batch_size = 4
    nb_epochs = 50
    nb_min_after_dequeue = 10
    nb_capacity = 3*nb_min_after_dequeue+nb_batch_size

    files = tf.train.match_filenames_once(tfrecord_path)
    file_queue = tf.train.string_input_producer(files,num_epochs=100)

    image, lable = preprocessing.restore_data(file_queue)
    # 图像预处理&图像提升
    image = preprocessing.crop_and_resize(image)
    # batch定义
    image_batch, label_batch = tf.train.shuffle_batch([image, lable],
                                                    batch_size=nb_batch_size,
                                                    min_after_dequeue=nb_min_after_dequeue,
                                                    capacity=nb_capacity)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(1):
            data,target = sess.run([image_batch,label_batch])
            print(data.shape)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()





