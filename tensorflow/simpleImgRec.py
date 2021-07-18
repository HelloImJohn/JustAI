import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import re, time
#import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

print("Tensorflow version " + tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#this line enables GPU usage
strategy = tf.distribute.MirroredStrategy()

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
#flower_photo/
#  daisy/
#  dandelion/
#  roses/
#  sunflowers/
#  tulips/

import pathlib
#dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
#data_dir = tf.keras.utils.get_file('/home/john/Documents/code/COMPANY/flower_photos', origin=dataset_url, untar=True)
#data_dir = tf.keras.utils.get_file('flower_photos', untar=True)
data_dir = pathlib.Path('/home/john/Documents/code/COMPANY/videoFolder/imgFromVideo')

print(data_dir)


#count images
image_count = len(list(data_dir.glob('*/*.png')))
print(image_count)


#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6024)])

#img = mpimg.imread('/videoFolder/imgFromVideo/candles/frame0.png')
#plt.imshow(img)
#plt.show()


# create a dataset
batch_size = 32
img_height = 331
img_width = 331
# define training data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
# define validation data
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# definition of classnames
class_names = train_ds.class_names
print(class_names)


# show some images of each class
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
  plt.show()


# check dimensionality of images for later tensor generation
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


### MAKE THE NETWORK (WORK) ###
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# preprocess to scale down 255 to values between 0 and 1
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

##
#normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
#image_batch, labels_batch = next(iter(normalized_ds))
#first_image = image_batch[0]
## Notice the pixels values are now in `[0,1]`.
#print(np.min(first_image), np.max(first_image)) 
## this is an alternative to the later solution that will refer to this one via a comment (you will see that this method is longer)

num_classes = 4

with strategy.scope():
  model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),   # this is an alternative to the rescaling comment from earlier
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=6
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# data augmentation      NOT AVAILABLE IN TF VERSION 2.4.X
#data_augmentation = keras.Sequential(
#  [
#    layers.experimental.preprocessing.RandomFlip("horizontal", 
#                                                 input_shape=(img_height, 
#                                                              img_width,
#                                                              3)),
#    layers.experimental.preprocessing.RandomRotation(0.1),
#    layers.experimental.preprocessing.RandomZoom(0.1),
#  ]
#)

#plt.figure(figsize=(10, 10))
#for images, _ in train_ds.take(1):
#  for i in range(9):
#    augmented_images = data_augmentation(images)
#    ax = plt.subplot(3, 3, i + 1)
#    plt.imshow(augmented_images[0].numpy().astype("uint8"))
#    plt.axis("off")
#  plt.show()

## make predictoin ##


pathOne = "file:///home/john/Documents/code/COMPANY/videoFolder/imgFromVideo/lamps/frame61.png"
pathTwo="file:///home/john/Downloads/592px-Red_sunflower.png"
img_path = tf.keras.utils.get_file('frame61.png', origin=pathOne)

img = keras.preprocessing.image.load_img(
    img_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
