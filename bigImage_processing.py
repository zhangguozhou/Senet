import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from PIL import Image

file_dir = './train/'

image_size = 256
image_height = 256
image_width = 256
img_channels = 3

class_num = 2
batch_size = 5
capacity = 50

def data_augmentation(batch):
    train_batch = np.array([])
    for i in range(len(batch)):
        j = np.array(batch[i])
        image = tf.read_file(j)
        img_data_jpg = tf.image.decode_jpeg(image)
        img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype = tf.uint8)
        img_data_jpg = tf.image.resize_images(img_data_jpg, [256,256],method = 2)
        #sess = tf.InteractiveSession()
        resized = np.asarray(img_data_jpg.eval(), dtype = 'uint8')
        train_batch = np.append(train_batch,resized)

    batch_x = train_batch.reshape([-1, image_size, image_size, img_channels])

    return batch_x

def get_file(file_dir):
    flaws=np.array([])
    label_flaw=np.array([])
    normals=np.array([])
    label_normal=np.array([])

    for file in os.listdir(file_dir + '/flaw'):
        flaws = np.append(flaws,file_dir + '/flaw/'+file)
        label_flaw = np.append(label_flaw, 0)
    for file in os.listdir(file_dir + '/normal'):
        normals = np.append(normals, file_dir + '/normal/'+file)
        label_normal = np.append(label_normal, 1)

    image_list = np.hstack((flaws, normals))
    label_list = np.hstack((label_flaw, label_normal))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = np.array(list(temp[:, 0]))
    label_list = np.array(list(temp[:, 1])).astype(np.float32)
    label_list = label_list.astype(np.int32)   

    if label_list[0] == 0:
        image_label = np.array([[1,0]])
    if label_list[0] == 1:
        image_label = np.array([[1,0]])

    for i in range(1, len(label_list)):
        if label_list[i] == 0:
            image_label = np.append(image_label, [[1, 0]], axis = 0)
        if label_list[i] == 1:
            image_label = np.append(image_label, [[0, 1]], axis = 0)
    print(image_label.shape)    

    test_index = np.random.randint(0,len(image_list),400)
    test_image=np.array([])
    test_label=np.array([])

    for i in range(400):
        test_image = np.append(test_image, image_list[test_index[i]])
        test_label = np.append(test_label, image_label[test_index[i]])

    return image_list, image_label, test_image, test_label


def get_batch(image, label, image_H, image_W, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image = tf.read_file(input_queue[0])

    image_batch, label_batch = tf.train.batch(
        [image, label], batch_size=batch_size, capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch


def color_preprocessing(x_train):
    x_train = x_train.astype('float32')
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    return x_train


#train_data = image_list.reshape([-1, img_channels, image_size, image_size])
#test_data = test_image.reshape([-1, img_channels, image_size, image_size])