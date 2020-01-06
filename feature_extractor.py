from __future__ import absolute_import, division, print_function, unicode_literals
from keras.models import load_model, Model
from keras.layers import *
from keras.applications.vgg16 import VGG16
import h5py
import time
import tensorflow as tf
import numpy as np
import pathlib

batch_size = 8192  # change based on GPU or CPU RAM
total_images = 73683

tf.debugging.set_log_device_placement(True)
with tf.device('/GPU:0'):  # change to CPU if GPU not available
    data_dir = pathlib.Path('./data/')  # The folder path where another folder of images are located
    # for an example image 0.jpg should be in /data/pics/0.jpg
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         target_size=(224, 224))
    model = VGG16(weights='imagenet', include_top=True)

    hf = h5py.File('./test_vgg16.h5', 'w')
    featextractor_model = Model(outputs=model.get_layer('fc2').output, inputs=model.input)
    train_ims = np.zeros((total_images, 4096))
    since = time.time()
    for i in range(0, np.ceil(total_images / batch_size)):
        image_batch, label_batch = next(train_data_gen)
        features = featextractor_model.predict(image_batch)
        train_ims[i * 8192:(i + 1) * 8192, :] = features

print("Time elapsed to extract features", (time.time() - since))
hf.create_dataset('train_ims', data=train_ims)
hf.close()

# Read file for debugging
with h5py.File("./test_vgg16.h5", 'r') as f:
    keys = list(f.keys())
    print(keys)
    train_ims = f[keys[0]].value
f.close()
print(type(train_ims))
print(train_ims.shape)
