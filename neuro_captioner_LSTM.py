# -*- coding: utf-8 -*-
"""Copy of neuro-captioner.ipynb

Automatically generated by Colaboratory.

"""

# !unzip -q /gdrive/My\ Drive/Neural/Final\ Project/data.zip -d data

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
print(tf.__version__)
print(tf.test.gpu_device_name())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from __future__ import absolute_import, division, print_function, unicode_literals
import h5py
import numpy as np
import requests
import matplotlib.pyplot as plt
import cv2
from sklearn.utils import shuffle
from PIL import Image
from io import BytesIO
from tensorflow import keras
from google.colab import drive
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import inception_v3
from keras.initializers import Constant
from tensorflow.keras.preprocessing import image
import cv2

drive.mount('/gdrive')
# %cd /gdrive/My Drive
# %cd /gdrive/My Drive/Neural/Final Project

class TrainGenerator(keras.utils.Sequence):
    def __init__(self, directory=".", batch_size=32, cnn_model=0, shuffle='True', dimensions=(224, 224, 3)):
        self.directory = directory
        self.batch_size = batch_size
        # self.N_features = N_features
        self.shuffle = shuffle
        self.dimensions = (dimensions[0], dimensions[1])

        with h5py.File(directory + "/eee443_project_dataset_train.h5", 'r') as f:
            keys = list(f.keys())
            all_cap = f[keys[0]][()]
            all_img_id = f[keys[1]][()]-1
            # train_ims = f[keys[2]][()] # remove
            f.close()

        # self.N_features = None
        # path = None
        # if cnn_model == 1:
        #     path = directory + "/vgg16.h5"
        #     self.N_features = 4096
        #     print("vgg16")
        # elif cnn_model == 2:
        #     path = directory + "/inceptionV3.h5"
        #     self.N_features = 2048
        #     print("inceptionV3")
        # elif cnn_model == 3:
        #     path = directory + "/new_topless_vgg16.h5"
        #     self.N_features = 512
        #     print("new_topless_vgg16")
        # elif cnn_model == 4:
        #     path = directory + "/new_topless_inceptionV3.h5"
        #     self.N_features = 2048
        #     print("new_topless_inceptionV3")
        # elif cnn_model == 0:  # remove
        #     self.image_features = train_ims  # remove
        #     self.N_features = 512  # remove
        #     print("moodle")
        
        # if cnn_model != 0:
        #     with h5py.File(path, 'r') as f:
        #         keys = list(f.keys())
        #         self.image_features = f[keys[0]][()]
        #         f.close()
            
        # self.image_features = (self.image_features - self.image_features.mean() ) / self.image_features.std() # normalize

        # remove data for images that are unavailable
        not_found = np.genfromtxt(directory + "/not_found.txt", delimiter=', ')
        not_found = not_found.astype(int)
        indexes = []
        for i in not_found:
            for j in np.where(all_img_id == i)[0]:
                indexes.append(j)
        self.all_cap = np.delete(all_cap, indexes, 0)
        self.all_img_id = np.delete(all_img_id, indexes)
        
        self.all_cap = self.all_cap[0 : int(0.8 * len(self.all_img_id)), :]
        self.all_img_id = self.all_img_id[0 : int(0.8 * len(self.all_img_id))]

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.all_img_id) / self.batch_size))

    def __getitem__(self, index):

        indices = range((index)*self.batch_size, (index+1)*self.batch_size)
        # input captions
        input_captions = np.zeros((self.batch_size, 16))
        input_captions[:,:] = self.all_cap[indices, 0:16]

        # output captions
        output_captions = np.zeros((self.batch_size,16,1004), dtype = int)
        
        for i in range(self.batch_size):
            for j in range(16):
                output_captions[i][j][self.all_cap[indices[i]][j+1]] = 1

        # # image features
        # img_features = np.zeros((self.batch_size, self.N_features))
        # for i in range(self.batch_size):
        #     img = self.image_features[self.all_img_id[indices[i]], :]
        #     img_features[i,:] = img

        # images
        imgs = np.zeros((self.batch_size, self.dimensions[0], self.dimensions[1], 3))
        for i in range(self.batch_size):
            imgs[i,:,:,:] = cv2.resize(cv2.imread("./data/data/" + str(self.all_img_id[indices[i]]) + ".jpg"), dsize=self.dimensions) / 255
            
        # position
        position = np.identity(16)
        position = np.reshape(position, (1,16,16))
        position = np.tile(position, (self.batch_size, 1, 1))

        inputs = [input_captions, imgs, position]
        return inputs, output_captions

    def on_epoch_end(self):  # To shuffle after each epoch
        permutation = np.random.permutation(len(self.all_img_id))
        if self.shuffle == True:
            self.all_cap = self.all_cap[permutation, :]
            self.all_img_id = self.all_img_id[permutation]

class TestGenerator(keras.utils.Sequence):
    def __init__(self, directory=".", batch_size=32, cnn_model=0, shuffle='True', dimensions=(224, 224, 3)):
        self.directory = directory
        self.batch_size = batch_size
        # self.N_features = N_features
        self.shuffle = shuffle
        self.dimensions = (dimensions[0], dimensions[1])

        with h5py.File(directory + "/eee443_project_dataset_train.h5", 'r') as f:
            keys = list(f.keys())
            all_cap = f[keys[0]][()]
            all_img_id = f[keys[1]][()]-1
            train_ims = f[keys[2]][()] # remove
            f.close()

        # self.N_features = None
        # path = None
        # if cnn_model == 1:
        #     path = directory + "/vgg16.h5"
        #     self.N_features = 4096
        #     print("vgg16")
        # elif cnn_model == 2:
        #     path = directory + "/inceptionV3.h5"
        #     self.N_features = 2048
        #     print("inceptionV3")
        # elif cnn_model == 3:
        #     path = directory + "/new_topless_vgg16.h5"
        #     self.N_features = 512
        #     print("new_topless_vgg16")
        # elif cnn_model == 4:
        #     path = directory + "/new_topless_inceptionV3.h5"
        #     self.N_features = 2048
        #     print("new_topless_inceptionV3")
        # elif cnn_model == 0:  # remove
        #     self.image_features = train_ims  # remove
        #     self.N_features = 512  # remove
        #     print("moodle")

        # if cnn_model != 0:
        #     with h5py.File(path, 'r') as f:
        #         keys = list(f.keys())
        #         self.image_features = f[keys[0]][()]
        #         f.close()
        
        # self.image_features = (self.image_features - self.image_features.mean() ) / self.image_features.std() # normalize

        # remove data for images that are unavailable
        not_found = np.genfromtxt(directory + "/not_found.txt", delimiter=', ')
        not_found = not_found.astype(int)
        indexes = []
        for i in not_found:
            for j in np.where(all_img_id == i)[0]:
                indexes.append(j)
        self.all_cap = np.delete(all_cap, indexes, 0)
        self.all_img_id = np.delete(all_img_id, indexes)

        self.N_valid = int(0.8 * len(self.all_img_id))

        self.all_cap = self.all_cap[self.N_valid : , :]
        self.all_img_id = self.all_img_id[self.N_valid : ]

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.all_img_id) / self.batch_size))

    def __getitem__(self, index):

        indices = range((index)*self.batch_size, (index+1)*self.batch_size)
        # input captions
        input_captions = np.zeros((self.batch_size, 16))
        input_captions[:,:] = self.all_cap[indices , 0:16]

        # output captions
        output_captions = np.zeros((self.batch_size,16,1004), dtype = int)
        for i in range(self.batch_size):
            for j in range(16):
                output_captions[i][j][self.all_cap[indices[i]][j+1]] = 1

        # # image features
        # img_features = np.zeros((self.batch_size, self.N_features))
        # for i in range(self.batch_size):
        #     img = self.image_features[self.all_img_id[indices[i]], :]
        #     img_features[i,:] = img

        # images
        imgs = np.zeros((self.batch_size, self.dimensions[0], self.dimensions[1], 3))
        for i in range(self.batch_size):
            imgs[i,:,:,:] = cv2.resize(cv2.imread("./data/data/" + str(self.all_img_id[indices[i]]) + ".jpg"), dsize=self.dimensions) / 255
            
        # position
        position = np.identity(16)
        position = np.reshape(position, (1,16,16))
        position = np.tile(position, (self.batch_size, 1, 1))

        inputs = [input_captions, imgs, position]
        return inputs, output_captions

    def on_epoch_end(self):  # To shuffle after each epoch
        permutation = np.random.permutation(len(self.all_img_id))
        if self.shuffle == True:
            self.all_cap = self.all_cap[permutation, :]
            self.all_img_id = self.all_img_id[permutation]

# Number of train data, validation data, image features, all words and time steps
# cnn_model    = 0  # from moodle
# cnn_model    = 1  # from VGG16
# cnn_model    = 2  # from inceptionV3
cnn_model    = 3  # from topless VGG16
# cnn_model    = 4  # from topless inceptionV3
N_words       = 1004
N_time_steps  = 16
N_batch       = 64

# train_directory = '/gdrive/My Drive'
# valid_directory = '/gdrive/My Drive'
train_directory = "/gdrive/My Drive//Neural/Final Project"
valid_directory = "/gdrive/My Drive//Neural/Final Project"

N_features = None
if cnn_model == 1:
    N_features = 4096
elif cnn_model == 2:
    N_features = 2048
elif cnn_model == 3:
    N_features = 512
    dimensions = (224, 224, 3)
elif cnn_model == 4:
    N_features = 2048
    dimensions = (299, 299, 3)
elif cnn_model == 0:  # remove
    N_features = 512  # remove

train_generator = TrainGenerator(directory=train_directory, batch_size=N_batch, cnn_model=cnn_model, dimensions=dimensions)
valid_generator = TestGenerator(directory=valid_directory, batch_size=N_batch, cnn_model=cnn_model, dimensions=dimensions)

# Replicate image features 16 times to create a time series.
if cnn_model == 4:
    images             = keras.Input(shape=dimensions, name = 'images')
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    model_cnn          = InceptionV3(weights='imagenet', include_top=False, pooling='avg')(images)
    pooling            = keras.layers.AveragePooling2D(pool_size=(8, 8))(model_cnn)
else:
    images             = keras.Input(shape=dimensions, name = 'images')
    from tensorflow.keras.applications.vgg16 import VGG16
    model_cnn          = VGG16(weights='imagenet', include_top=False, pooling='avg')(images)

replicate_func     = lambda x: tf.tile(tf.reshape(x, (tf.shape(x)[0], 1, N_features)), (1, N_time_steps, 1))
img_features       = keras.layers.Lambda(replicate_func)(model_cnn)

caption_words      = keras.Input(shape = (N_time_steps), name = 'Caption_Words')

# Using Google's word2vec model
embed_weight       = np.load('/gdrive/My Drive//Neural/Final Project/embed_words.npy')
embedding_layer    = keras.layers.Embedding(N_words, 300, embeddings_initializer=Constant(embed_weight), trainable=True, name = 'Word_Embedding')(caption_words)

position           = keras.Input(shape=(N_time_steps,N_time_steps), name = 'Position')

concat             = keras.layers.Concatenate(axis=2, name = 'Concetenation')([embedding_layer, img_features, position])

lstm        = keras.layers.LSTM(256, dropout=0.5, return_sequences = True)(concat)
#drop_out   = keras.layers.Dropout(0.5)(lstm)
norm        = keras.layers.BatchNormalization()(lstm)
soft_max    = keras.layers.Dense(N_words, activation='softmax')(norm)
model       = keras.Model(inputs=[caption_words, images, position], outputs=soft_max)
model.summary()

# from tensorflow.keras.applications.vgg16 import VGG16
# model_cnn          = VGG16(weights='imagenet', include_top=False)
# model = keras.Model(model_cnn.input, model_cnn.output)
with tf.device('/GPU:0'):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_generator, validation_data = valid_generator, epochs=5, use_multiprocessing=True, workers=2)
    model.save('inception_ownEmbed_5epoch.h5')

"""30 minutes for just CNN VGG16 per epoch

39 min
"""

model = keras.models.load_model('inception_ownEmbed_5epoch.h5')

directory = train_directory
with h5py.File(directory + "/eee443_project_dataset_train.h5", 'r') as f:
    keys = list(f.keys())
    all_cap = f[keys[0]][()]
    all_img_id = f[keys[1]][()]-1
    train_ims = f[keys[2]][()] # remove
    all_url = f[keys[3]][()]
    word_code = f[keys[4]][()]
    f.close()

# path = None
# if cnn_model == 1:
#     path = directory + "/vgg16.h5"
#     N_features == 4096
#     print("vgg16")
# elif cnn_model == 2:
#     path = directory + "/inceptionV3.h5"
#     N_features == 2048
#     print("inceptionV3")
# elif cnn_model == 3:
#     path = directory + "/new_topless_vgg16.h5"
#     N_features == 512
#     print("new_topless_vgg16")
# elif cnn_model == 4:
#     path = directory + "/new_topless_inceptionV3.h5"
#     N_features == 2048
#     print("new_topless_inceptionV3")
# elif cnn_model == 0:  # remove
#     image_features = train_ims  # remove
#     N_features = 512  # remove
#     print("moodle")

# if cnn_model != 0:
#     with h5py.File(path, 'r') as f:
#         keys = list(f.keys())
#         image_features = f[keys[0]][()]
#         f.close()

# image_features = (image_features - image_features.mean() ) / image_features.std() # normalize

if N_features == 0:  # remove
    image_features = train_ims  # remove

# remove data for images that are unavailable
not_found = np.genfromtxt(directory + "/not_found.txt", delimiter=', ')
not_found = not_found.astype(int)
indexes = []
for i in not_found:
    for j in np.where(all_img_id == i)[0]:
        indexes.append(j)
all_cap = np.delete(all_cap, indexes, 0)
all_img_id = np.delete(all_img_id, indexes)

# position
position = np.identity(16)
position = np.reshape(position, (1,16,16))

N_predict = 50000
cur_caption = np.zeros((1,16))
cur_caption[0,0] = 1

temp_range = range(all_img_id[N_predict],all_img_id[N_predict]+1)

img = np.reshape(cv2.resize(cv2.imread("./data/data/" + str(all_img_id[N_predict]) + ".jpg"), dsize=(dimensions[0], dimensions[1])) / 255, (1, dimensions[0], dimensions[1], dimensions[2]))

for i in range(15):
    prediction = model.predict(x=[cur_caption, img, position])
    word_pre = np.argmax(prediction, axis=2)
    cur_caption[0,i+1] = word_pre[0][i]

def get_word(word_code, index):
    for name in word_code.dtype.names:
        if word_code[name] == index:
            return name


def get_sentence(word_code, index_array):
    sentence = []
    for i in index_array:
        sentence.append(get_word(word_code, i))
    return sentence


def get_all_captions(word_code, train_cap, train_imid, image_number):
    cap_list = np.where(train_imid == image_number)  # get the list of captions of the image
    captions = []
    for x in cap_list[0]:
        captions.append(get_sentence(word_code, train_cap[x, :]))
    return captions

response = requests.get(all_url[all_img_id[N_predict]])
img = Image.open(BytesIO(response.content))
plt.imshow(img)
print(get_sentence(word_code, cur_caption[0,:]))
print(get_sentence(word_code, all_cap[N_predict,:]))
print('Image number:' + str(all_img_id[N_predict]))