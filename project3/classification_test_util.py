from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from scipy import ndimage
from skimage import transform
import numpy as np


model = load_model('fine_tuned_model.h5')

image_name = None
while image_name is None:
    image_name = input('Enter the test image file name: ')
    try:
        test_image = ndimage.imread(image_name)
        break
    except NameError:
        image_name = None
        print("invalid")

y = np.expand_dims(transform.resize(test_image, (150, 150)), axis=0)

# print(y.shape)

result = model.predict(y)
# print(result)

if result[0][0] < 0.5:
    print("cat")
else:
    print("dog")

import gc; gc.collect()
