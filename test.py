from Arguement_and_read import crop_and_resize
from scipy import misc
from matplotlib import pyplot as plt
import tensorflow as tf

img = misc.imread('data/car1/0a648e49-3f71-4857-bbb1-1ff99659d842.jpg')

with tf.Session() as sess:
    img = crop_and_resize(img)
    plt.imshow(img)
    plt.show()