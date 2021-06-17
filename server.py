from flask import Flask, request #to get data
import numpy as np
import tensorflow as tf

app = Flask(__name__)
from tensorflow import keras

#Flask => webSerivce (small pakeage python)

#192.168.1.18/go_ml
@app.route('/go_ml', methods=['POST']) #route get post #route == execute to function
def ML():
    try:

        imagefile = request.files['image'] #from java
        filename = imagefile.filename #image.jpg (just name)
        imagefile.save(filename)#store image (actually)
        return fun(filename) #to java
    except:
        return "Plant___Not Found"



def fun(filename):
    img_height = 250 #pic size (250*250)
    img_width = 250
    image_size = (img_height, img_width)
    model = keras.models.load_model('model.h5') #load model
    img = keras.preprocessing.image.load_img(
        filename, target_size=image_size)

    img_array = keras.preprocessing.image.img_to_array(img)#convert to array
    img_array = tf.expand_dims(img_array, 0)  #optimaization to array img

    predictions = model.predict(img_array) #get prediction
    return class_names[np.argmax(predictions[0])]#return label prediction

#np.argmax(predictions[0]) => put prediction return index from class name

class_names = ['Apple___Apple_scab',
               'Apple___Black_rot',
               'Apple___Cedar_apple_rust',
               'Apple___healthy',
               'Blueberry___healthy',
               'Cherry_(including_sour)___Powdery_mildew',
               'Cherry_(including_sour)___healthy',
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)___Common_rust_',
               'Corn_(maize)___Northern_Leaf_Blight',
               'Corn_(maize)___healthy',
               'Grape___Black_rot',
               'Grape___Esca_(Black_Measles)',
               'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
               'Grape___healthy',
               'Orange___Haunglongbing_(Citrus_greening)',
               'Peach___Bacterial_spot',
               'Peach___healthy',
               'Pepper,_bell___Bacterial_spot',
               'Pepper,_bell___healthy',
               'Potato___Early_blight',
               'Potato___Late_blight',
               'Potato___healthy',
               'Raspberry___healthy',
               'Soybean___healthy',
               'Squash___Powdery_mildew',
               'Strawberry___Leaf_scorch',
               'Strawberry___healthy',
               'Tomato___Bacterial_spot',
               'Tomato___Early_blight',
               'Tomato___Late_blight',
               'Tomato___Leaf_Mold',
               'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite',
               'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
               'Tomato___Tomato_mosaic_virus',
               'Tomato___healthy']



app.run(host='192.168.1.106', port=80, debug=True)#response run flask in pc
                                                #run api flask