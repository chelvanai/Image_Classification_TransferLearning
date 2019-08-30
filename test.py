import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import keras as k

# given the model load directory
model = tf.keras.models.load_model('./Model/weight.h5', custom_objects={'KerasLayer':hub.KerasLayer})
model.layers[0].input_shape

image_generator = k.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str('./fonts'), target_size=(224,224)) # image folder directory for the name of the image

# give want to predict image path
url = './test_images/x.png'
print("Given images:- ",url.split(('/'))[-1])

img = tf.keras.preprocessing.image.load_img(url, target_size=(224, 224))

x = k.preprocessing.image.img_to_array(img)/255

x = np.expand_dims(x, axis=0)

result = model.predict(x)

ans = np.argmax(result)

class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])
print("Predicated value",class_names[ans])
