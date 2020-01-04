# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 05:00:26 2019

@author: Ahmed Imam Shah
"""

from __future__ import absolute_import, division, print_function, unicode_literals
# import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
# import pylab as py
import h5py
# import math
from PIL import Image
import requests
from io import BytesIO

import urllib.request

# def url_is_alive(url):
#     """
#     Checks that a given URL is reachable.
#     :param url: A URL
#     :rtype: bool
#     """
#     request = urllib.request.Request(url)
#     request.get_method = lambda: 'HEAD'
#
#     try:
#         urllib.request.urlopen(request)
#         return True
#     except urllib.request.HTTPError:
#         return False

with h5py.File("eee443_project_dataset_train.h5", 'r') as f:
    keys = list(f.keys())
    print(keys)
    # train_cap  = f[keys[0]].value
    # train_imid  = f[keys[1]].value
    # train_ims = f[keys[2]].value
    # train_url = f[keys[3]].value
    # word_code = f[keys[4]].value
    f.close()

# for image_path in range(0,len(train_url)):
#     # if url_is_alive(train_url[image_path]):
#     try:
#         response = requests.get(train_url[image_path])
#         img = Image.open(BytesIO(response.content))
#         imnp = np.asarray(img)
#         img.save(str(image_path)+".jpg")
#     except Exception as e:
#         print("image " + str(image_path + 1) + "is not available")
#     #plt.figure()
#     #plt.imshow(img)
