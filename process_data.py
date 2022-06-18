from PIL import Image
import numpy as np
import tensoflow as tf




def process_image(image,Image_Size=(128,128)):
    '''
    Make an image ready-to-use by VGG19
    '''
    im=Image.open(image)
    im=im.resize(Image_Size)
    im=np.expand_dims(im,axis=0)
    im=np.array(im)
    im=im/255

    return image

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
    prediction = model.predict(image)


    return prediction, results[prediction]