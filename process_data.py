import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array




def process_image(image,Image_Size=(128,128)):
    '''
    Make an image ready-to-use by VGG19
    '''
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model

    return image/255.0

def predict_class(image):
    model = tf.keras.models.load_model('./models/model1_catsVSdogs.h5', compile=False)
    '''
    Predict and render the class of a given image 
    '''
    results={
        0:'cat',
        1:'dog'
    }
    # predict the probability across all output classes
    probability = model.predict(image)
    prediction = np.argmax(probability, axis=-1)

    return results[prediction[0]],probability[0][0]