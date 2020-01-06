import h5py
import numpy as np


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


with h5py.File("eee443_project_dataset_train.h5", 'r') as f:
    keys = list(f.keys())
    train_cap = f[keys[0]].value
    train_imid = f[keys[1]].value
    train_ims = f[keys[2]].value
    train_url = f[keys[3]].value
    word_code = f[keys[4]].value
    f.close()

# example usage
# print(get_sentence(word_code, train_cap[15, :]))  # get the 15th caption
#
# cap_list = np.where(train_imid == 20)  # get the list of captions of the 20th image
# print(get_sentence(word_code, train_cap[cap_list[0][2], :]))  # get the 3rd caption of the 20th image
#
# print(get_all_captions(word_code, train_cap, train_imid, 33))  # get all the captions of the 33rd image
