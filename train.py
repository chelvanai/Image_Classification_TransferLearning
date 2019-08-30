import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import keras as k

# using tesnorflow 1.4
print('tenso  rflow version', tf.__version__)
print('keras version', tf.keras.__version__)

data_root = './fonts'  # Use image category folder here [ animal images folder = {sub folder(cat,dog,cow,lion)}

image_shape = (224,224)

image_generator = k.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root), target_size=image_shape)

feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,input_shape=(224,224,3))
feature_extractor_layer.trainable = False

model = tf.keras.Sequential([
  feature_extractor_layer,
  tf.keras.layers.Dense(image_data.num_classes, activation='softmax')
])

model.summary()

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss='categorical_crossentropy',
  metrics=['acc'])


class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)

batch_stats_callback = CollectBatchStats()

model.fit(image_data, epochs=2,steps_per_epoch=steps_per_epoch,callbacks = [batch_stats_callback])

model.save('weight.h5')



