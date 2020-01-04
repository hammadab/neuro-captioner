import numpy as np
import h5py
from PIL import Image
import requests
from io import BytesIO
import urllib.request
import multiprocessing

with h5py.File("eee443_project_dataset_train.h5", 'r') as f:
    keys = list(f.keys())
    train_cap = f[keys[0]].value
    train_imid = f[keys[1]].value
    train_ims = f[keys[2]].value
    train_url = f[keys[3]].value
    word_code = f[keys[4]].value
    f.close()

not_found = []


def blah(image_path):
    try:
        response = requests.get(train_url[image_path])
        img = Image.open(BytesIO(response.content))
        imnp = np.asarray(img)
        img.save("./" + str(image_path) + ".jpg")
        return -1
    except Exception as e:
        return image_path


image_path = list(range(0, len(train_url)))
if __name__ == '__main__':
    with multiprocessing.Pool(1000) as p:
        not_found = p.map(blah, image_path)
unavailable = []
for i, x in enumerate(not_found):
    if x != -1:
        unavailable.append(x)
print(unavailable)
