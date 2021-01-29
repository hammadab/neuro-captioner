# neuro-captioner
This project takes an image and generates a suitable caption. It uses one of the two CNN to extract features of the image then feeds it into an LSTM. The LSTM then generates a sentence, word by word. This project is built on Python 3.7.4 using TensorFlow 1.14. Inception_v3 and VGG16 are the two pre-trianed CNN used in this project. It uses the Flickr30k dataset to train and test. Each image is resized to 299 x 299 pixels for Inception_v3 and 224 x 224 pixels for VGG16.

Here is an example:

![Image 11365](/11365.png)

Caption: a street corner with a light in front of it
