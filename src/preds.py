from process_data import *

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


image = load_img('../input/cat_1.jpeg', target_size=(128, 128))

image = process_image(image)

results,probability = predict_class(image)

print(results,probability)

