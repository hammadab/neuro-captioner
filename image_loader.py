import h5py
from PIL import Image
import requests
from io import BytesIO
import multiprocessing


def image_downloader(index):
    try:
        response = requests.get(train_url[index])
        img = Image.open(BytesIO(response.content))
        img.save("./" + str(index) + ".jpg")
        return -1
    except Exception:
        return index


with h5py.File("eee443_project_dataset_train.h5", 'r') as f:
    keys = list(f.keys())
    train_cap = f[keys[0]].value
    train_imid = f[keys[1]].value
    train_ims = f[keys[2]].value
    train_url = f[keys[3]].value
    word_code = f[keys[4]].value
    f.close()

temp_not_found = []
threads_count = 1000  # adjust based on your CPU
image_path = list(range(0, len(train_url)))
if __name__ == '__main__':
    with multiprocessing.Pool(threads_count) as p:
        temp_not_found = p.map(image_downloader, image_path)
not_found = []
for i, x in enumerate(temp_not_found):
    if x != -1:
        not_found.append(x)
print(not_found)
